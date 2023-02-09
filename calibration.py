import pandas as pd
from scipy.stats import norm, kurtosis, skew
from statistics import mean, variance
import scipy.integrate as integrate
from scipy.linalg import solve
from scipy import optimize, special
import numpy as np
import math
import os
import time

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, './Data10.xlsx')
dataEquity = pd.read_excel(filename, sheet_name='Mod Market Cap') 
dataEquity = dataEquity.set_index('Dates').loc['2019-10-28':'2020-10-13']
dataDebt = pd.read_excel(filename, sheet_name='Gross Debt').dropna()

class calibrationModels():

    def __init__(self, companyEquityTicker = 'CRH LN Equity', timePeriod = 252, horizon = 1,tolerance = 10e-5, riskNeutral = False):
        
        self.companyEquityTicker_       = companyEquityTicker
        self.tolerance_                 = tolerance
        self.timePeriod_                = timePeriod #Correspond au nombre de jours de la période étudiée
        self.horizon_                   = horizon
        self.relativeTime_              = 0 #Correspond au temps relatif où on effectue le calcul
        self.riskFIR_                   = 0 #Risk-free interest rate
        self.riskNeutral_               = riskNeutral

        ### Modèle Merton
        self.sigma_A_                   = 0
        self.sigma_E_                   = 0
        self.sigma_A_history_           = [-100]

        ### Modèle NegIG and NegGamma 
        self.rho_                       = 0
        self.lambda_                    = 0
        self.mu_                        = 0

        self.rho_history_               = [-100]
        self.lambda_history_            = [-100]
        self.mu_history_                = [-100]

        ### Variance Gamma Physical parameters
        self.nu_                        = 0
        self.sigma_                     = 0
        self.theta_                     = 0
        self.nu_history_                = [-100]
        self.sigma_history_             = [-100]
        self.theta_history_             = [-100]

        ### Variance Gamma Risk neutral parameters
        self.nuRN_                      = 0
        self.sigmaRN_                   = 0
        self.thetaRN_                   = 0
        self.nuRN_history_              = [-100]
        self.sigmaRN_history_           = [-100]
        self.thetaRN_history_           = [-100]

        ## Bilateral Gamma physical parameters
        self.alphaP_                    = 0
        self.alphaM_                    = 0
        self.lambdaP_                   = 0
        self.lambdaM_                   = 0
        self.alphaP_history_            = [-100]
        self.alphaM_history_            = [-100]
        self.lambdaP_history_           = [-100]
        self.lambdaM_history_           = [-100]

        ## Bilateral Gamma risk neutral parameters
        self.alphaP_RN_                 = 0
        self.alphaM_RN_                 = 0
        self.lambdaP_RN_                = 0
        self.lambdaM_RN_                = 0
        self.alphaP_RN_history_         = [-100]
        self.alphaM_RN_history_         = [-100]
        self.lambdaP_RN_history_        = [-100]
        self.lambdaM_RN_history_        = [-100]

        ## Central moments
        self.mean_                      = 0
        self.variance_                  = 0
        self.kurtosis_                  = 0
        self.skewness_                  = 0

        self.asset_values_              = [] #Correspond à V_A
        self.equity_value_              = 0  #Correspond à V_E

        self.nombreIterations_          = 0
        self.companyDebt_               = dataDebt[[self.companyEquityTicker_]].iloc[0,0]
        self.companyEquityListValues_   = dataEquity[[self.companyEquityTicker_]].iloc[:,0]

    ##################################################################################################################
    ################################### Fonctions de répartition et densités #########################################
    ##################################################################################################################
    
    def cumulativeGaussianDistribution(self,x):
        return norm.cdf(x)

    def gamma(self,x):
        return special.gamma(x)

    def lowerIncompleteGamma(self, a, x):
        return self.gamma(a)*special.gammainc(a, x)

    def upperIncompleteGamma(self, a, x):
        return self.gamma(a)*special.gammaincc(a, x)

    def varianceGammaPDF(self, x):
        "VarGamma probability density function in a point x"

        if not self.riskNeutral_:
            temp1 = 2.0 / (self.sigma_*(2.0*np.pi)**0.5*self.nu_**(self.horizon_/self.nu_)*self.gamma(self.horizon_/self.nu_) )
            temp2 = ((2*self.sigma_**2/self.nu_+self.theta_**2)**0.5)**(0.5-self.horizon_/self.nu_)
            temp3 = np.exp(self.theta_ * x/self.sigma_**2) * abs(x)**(self.horizon_/self.nu_ - 0.5)
            temp4 = special.kv(self.horizon_/self.nu_ - 0.5, abs(x)*(2*self.sigma_**2/self.nu_+self.theta_**2)**0.5/self.sigma_**2)
            return temp1*temp2*temp3*temp4
        else:
            temp1 = 2.0 / (self.sigmaRN_*(2.0*np.pi)**0.5*self.nuRN_**(self.horizon_/self.nuRN_)*self.gamma(self.horizon_/self.nuRN_) )
            temp2 = ((2*self.sigmaRN_**2/self.nuRN_+self.thetaRN_**2)**0.5)**(0.5-self.horizon_/self.nuRN_)
            temp3 = np.exp(self.thetaRN_ * x/self.sigmaRN_**2) * abs(x)**(self.horizon_/self.nuRN_ - 0.5)
            temp4 = special.kv(self.horizon_/self.nuRN_ - 0.5, abs(x)*(2*self.sigmaRN_**2/self.nuRN_+self.thetaRN_**2)**0.5/self.sigmaRN_**2)
            return temp1*temp2*temp3*temp4
    
    def varianceGammaCDF(self, x):
        "VarGamma cumulative distribution function in a point x"

        return integrate.quad(lambda x: self.varianceGammaPDF(x), -100, x)[0]

    def whittakerFunction(self, x):

        if not self.riskNeutral_:
            if x > 0:
                #lambdaParam_    = 0.5 * (self.alphaP_ - self.alphaM_) * self.horizon_ #self.relativeTime_ ou tau ???
                lambdaParam_    = 0.5 * (self.alphaP_ - self.alphaM_) * self.relativeTime_
            else:
                #lambdaParam_    = 0.5 * (self.alphaM_ - self.alphaP_) * self.horizon_ #self.relativeTime_ ou tau ???
                lambdaParam_    = 0.5 * (self.alphaM_ - self.alphaP_) * self.relativeTime_
            
            #muParam_        = 0.5 * ((self.alphaP_ + self.alphaM_) * self.horizon_ - 1) #self.relativeTime_ ou tau ???
            muParam_        = 0.5 * ((self.alphaP_ + self.alphaM_) * self.relativeTime_ - 1)

            def integrande(t):
                return t**(muParam_ - lambdaParam_ - 0.5) * np.exp(-t) * (1 + t/np.abs(x))**(muParam_ + lambdaParam_ - 0.5)

            leftPart_       = np.abs(x)**lambdaParam_ * np.exp(- 0.5 * np.abs(x)) / self.gamma(muParam_ - lambdaParam_ + 0.5)
            rightPart_      = integrate.quad(lambda t: integrande(t), 0, np.inf)[0]

            return leftPart_ * rightPart_
        
        else:
            if x > 0:
                lambdaParam_    = 0.5 *  (self.alphaP_RN_ - self.alphaM_RN_) * self.relativeTime_ #self.relativeTime_ ou tau ???
            else:
                lambdaParam_    = 0.5 *  (self.alphaM_RN_ - self.alphaP_RN_) * self.relativeTime_ #self.relativeTime_ ou tau ???
            
            muParam_        = 0.5 * ((self.alphaP_RN_ + self.alphaM_RN_) * self.relativeTime_ - 1) #self.relativeTime_ ou tau ???

            def integrande(t):
                return t**(muParam_ - lambdaParam_ - 0.5) * np.exp(-t) * (1 + t/np.abs(x))**(muParam_ + lambdaParam_ - 0.5)

            leftPart_       = np.abs(x)**lambdaParam_ * np.exp(- 0.5 * np.abs(x)) / self.gamma(muParam_ - lambdaParam_ + 0.5)
            rightPart_      = integrate.quad(lambda t: integrande(t), 0, np.inf)[0]

            return leftPart_ * rightPart_ 

    def bilateralGammaPDF(self, x):

        t   = self.relativeTime_ ## ou self.relativeTime ou tau???

        if not self.riskNeutral_:
            if x > 0:
                aPlus   = self.alphaP_
                aMoins  = self.alphaM_
                lPlus   = self.lambdaP_
                lMoins  = self.lambdaM_
            else:
                aPlus   = self.alphaM_
                aMoins  = self.alphaP_
                lPlus   = self.lambdaM_
                lMoins  = self.lambdaP_

            num1    = lPlus**(aPlus * t) * lMoins**(aMoins * t) * np.abs(x)**(0.5 * (aPlus + aMoins) * t - 1) * np.exp(- 0.5 * (lPlus - lMoins) * np.abs(x))
            num2    = self.whittakerFunction((lPlus + lMoins) * x)
            den     = ((lPlus + lMoins)**(0.5 * (aPlus + aMoins) * t)) * self.gamma(aPlus * t)

            return num1 * num2 / den

        else:
            if x > 0:
                aPlus   = self.alphaP_RN_
                aMoins  = self.alphaM_RN_
                lPlus   = self.lambdaP_RN_
                lMoins  = self.lambdaM_RN_
            else:
                aPlus   = self.alphaM_RN_
                aMoins  = self.alphaP_RN_
                lPlus   = self.lambdaM_RN_
                lMoins  = self.lambdaP_RN_

            num1    = lPlus**(aPlus * t) * lMoins**(aMoins * t) * np.abs(x)**(0.5 * (aPlus + aMoins) * t - 1) * np.exp(- 0.5 * (lPlus - lMoins) * np.abs(x))
            num2    = self.whittakerFunction((lPlus + lMoins) * x)
            den     = (lPlus + lMoins)**(0.5 * (aPlus + aMoins) * t) * self.gamma(aPlus * t)

            return num1 * num2 / den  

    def bilateralGammaCDF(self, x):

        return integrate.quad(lambda x: self.bilateralGammaPDF(x), -10, x)[0] ### -100 suffisant ??

    ##################################################################################################################
    ########################################### Les différents modèles ###############################################
    ##################################################################################################################

    def BlackScholesMertonModel(self):

        def d1(x):
            return ( (np.log(x/self.companyDebt_)) + (self.riskFIR_ + 0.5*self.sigma_A_**2)*self.relativeTime_ ) / (self.sigma_A_ * np.sqrt(self.relativeTime_))
        
        def d2(x):
            return d1(x) - self.sigma_A_ * np.sqrt(self.relativeTime_)

        def modelMerton(x):
            leftPart    = x*self.cumulativeGaussianDistribution(d1(x))
            rightPart   = self.companyDebt_*np.exp(-self.riskFIR_ * self.relativeTime_)*self.cumulativeGaussianDistribution(d2(x))
            return leftPart - rightPart - self.equity_value_

        #On calcule la première valeur de sigma_E que l'on utilise comme valeur initiale de sigma_A
        self.sigma_E_ = np.std(np.diff(np.log(self.companyEquityListValues_), n = 1))*np.sqrt(self.timePeriod_)
        
        #On l'ajoute à l'historique des sigma_A
        self.sigma_A_history_.append(self.sigma_E_)

        while np.abs(self.sigma_A_history_[-1] - self.sigma_A_history_[-2]) > self.tolerance_:

            self.asset_values_      = []
            self.sigma_A_           = self.sigma_A_history_[-1] #On prend la dernière valeur estimée de sigma_A dans la boucle précédente

            for day in range(self.companyEquityListValues_.shape[0]):

                self.relativeTime_ = self.horizon_ + (self.timePeriod_ - day - 1)/252
                self.equity_value_ = self.companyEquityListValues_[day]
                self.asset_values_.append(optimize.newton(modelMerton,self.companyDebt_))

            self.sigma_A_history_.append(np.std(np.diff(np.log(self.asset_values_),n=1))*np.sqrt(self.timePeriod_))
            print(f"A l'itération {self.nombreIterations_} sigma = {round(self.sigma_A_history_[-1]*100,2)}% et VA = {self.asset_values_[-1]}")
            self.nombreIterations_  += 1
        
        self.sigma_A_               = self.sigma_A_history_[-1]
        mertonDistanceToDefault     = d2(self.asset_values_[-1])
        mertonDefaultProbability    = (1 - self.cumulativeGaussianDistribution(mertonDistanceToDefault))*100
        
        return self.nombreIterations_, self.asset_values_[-1], self.sigma_A_, mertonDistanceToDefault, mertonDefaultProbability
    
    def NegativeGammaModel(self):

        def kG(x):
            return np.log(x/self.companyDebt_) + (self.riskFIR_ + self.rho_ * np.log(1 + 1/self.lambda_)) * self.relativeTime_

        def parameterRho(kurtosis):
            return 6/kurtosis

        def parameterLambda(variance, rho):
            return np.sqrt(rho/variance)

        def modelNegGamma(x):
            if kG(x) <= 0:
                return -self.equity_value_
                
            else:
                rhoXtime    = self.rho_*self.relativeTime_
                leftPart    = x*self.lowerIncompleteGamma(rhoXtime,(1+self.lambda_)*kG(x))/self.gamma(rhoXtime)
                rightPart   = self.companyDebt_*np.exp(-self.riskFIR_*self.relativeTime_)*self.lowerIncompleteGamma(rhoXtime, self.lambda_*kG(x))/self.gamma(rhoXtime)
                return leftPart - rightPart - self.equity_value_

        
        #initialisation des paramètres
        equityRelativeValue     = np.diff(np.log(self.companyEquityListValues_), n = 1)
        
        self.rho_               = parameterRho(kurtosis(equityRelativeValue, fisher = False))
        self.lambda_            = parameterLambda(variance(equityRelativeValue)*self.timePeriod_, self.rho_)

        self.rho_history_.append(self.rho_)
        self.lambda_history_.append(self.lambda_)

        while (np.abs(self.rho_history_[-1] - self.rho_history_[-2]) > self.tolerance_) or (np.abs(self.lambda_history_[-1] - self.lambda_history_[-2]) > self.tolerance_):
                
            self.asset_values_  = []
            self.rho_           = self.rho_history_[-1]
            self.lambda_        = self.lambda_history_[-1]

            for day in range(self.companyEquityListValues_.shape[0]):

                self.relativeTime_ = self.horizon_ + (self.timePeriod_ - day - 1)/252
                self.equity_value_ = self.companyEquityListValues_[day]
                self.asset_values_.append(optimize.newton(modelNegGamma,self.companyDebt_))

            assetRelativeValue  = np.diff(np.log(self.asset_values_), n = 1)
            kurtosis_           = kurtosis(assetRelativeValue, fisher = False)
            variance_           = variance(assetRelativeValue)*252

            self.rho_history_.append(parameterRho(kurtosis_))
            self.lambda_history_.append(parameterLambda(variance_, self.rho_history_[-1]))
            print(f"A l'itération {self.nombreIterations_} rho = {round(self.rho_history_[-1],3)} et lambda = {round(self.lambda_history_[-1],3)} et VA = {self.asset_values_[-1]}")
            self.nombreIterations_ += 1

        self.rho_           = self.rho_history_[-1]
        self.lambda_        = self.lambda_history_[-1]

        negGammaDistanceToDefault   = kG(self.asset_values_[-1])
        negGammaDefaultprobability  = 1 if negGammaDistanceToDefault <= 0 else self.upperIncompleteGamma(self.rho_*self.relativeTime_  ,self.lambda_*negGammaDistanceToDefault)/self.gamma(self.rho_ * self.relativeTime_)*100
            
        return self.nombreIterations_, self.asset_values_[-1], self.rho_, self.lambda_, negGammaDistanceToDefault, negGammaDefaultprobability

    def NegInvGaussianModel(self):

        def kI(x):
            return np.log(x/self.companyDebt_) + (self.riskFIR_ + (self.lambda_/self.mu_)*(np.sqrt(1 + 2*(self.mu_)**2/self.lambda_) - 1))*self.relativeTime_

        def parameterLambda(variance, kurtosis):
            return (15/kurtosis)*np.sqrt(15*variance/kurtosis)

        def parameterMu(variance, kurtosis):
            return np.sqrt(15*variance/kurtosis)

        def functionPhi(x, lambd):
            leftPart    = self.cumulativeGaussianDistribution(np.sqrt(lambd*(self.relativeTime_)**2/x)*(x/(self.mu_*self.relativeTime_)-1))
            rightPart   = np.exp(2*lambd*self.relativeTime_/self.mu_) * self.cumulativeGaussianDistribution( - np.sqrt(lambd*(self.relativeTime_)**2/x)*(x/(self.mu_*self.relativeTime_)+1))
            
            return leftPart + rightPart

        def defaultProbability(distDefault):
            leftPart    = self.cumulativeGaussianDistribution(-np.sqrt(self.lambda_*(self.relativeTime_)**2/(distDefault))*(distDefault/(self.mu_*self.relativeTime_)-1))
            rightPart   = np.exp(2*self.lambda_*self.relativeTime_/self.mu_) * self.cumulativeGaussianDistribution(-np.sqrt(self.lambda_*(self.relativeTime_)**2/(distDefault))*(distDefault/(self.mu_*self.relativeTime_)+1))

            return leftPart - rightPart
        
        def modelNegIG(x):
            if kI(x) <= 0:
                return - self.equity_value_
            
            else:
                leftPart    = x * functionPhi(kI(x)*np.sqrt(1+2*(self.mu_)**2/self.lambda_),self.lambda_*np.sqrt(1+2*(self.mu_)**2/self.lambda_))
                rightPart   = self.companyDebt_*np.exp(-self.riskFIR_*self.relativeTime_) * functionPhi(kI(x), self.lambda_)

                return leftPart - rightPart - self.equity_value_
         
        #initialisation des paramètres

        equityRelativeValue     = np.diff(np.log(self.companyEquityListValues_), n = 1)

        self.lambda_            = parameterLambda(variance(equityRelativeValue)*self.timePeriod_, kurtosis(equityRelativeValue, fisher = False))
        self.mu_                = parameterMu(variance(equityRelativeValue)*self.timePeriod_, kurtosis(equityRelativeValue, fisher = False))

        self.lambda_history_.append(self.lambda_)
        self.mu_history_.append(self.mu_)

        while (np.abs(self.lambda_history_[-1] - self.lambda_history_[-2]) > self.tolerance_) or (np.abs(self.mu_history_[-1] - self.mu_history_[-2]) > self.tolerance_):

            self.asset_values_  = []
            self.lambda_        = self.lambda_history_[-1]
            self.mu_            = self.mu_history_[-1]
            
            for day in range(self.companyEquityListValues_.shape[0]):

                self.relativeTime_ = self.horizon_ + (self.timePeriod_ - day - 1)/self.timePeriod_
                self.equity_value_ = self.companyEquityListValues_[day]
                self.asset_values_.append(optimize.newton(modelNegIG,self.companyDebt_))

            assetRelativeValue = np.diff(np.log(self.asset_values_), n = 1)
            kurtosis_           = kurtosis(assetRelativeValue, fisher = False)
            variance_           = variance(assetRelativeValue)*self.timePeriod_

            self.lambda_history_.append(parameterLambda(variance_, kurtosis_))
            self.mu_history_.append(parameterMu(variance_, kurtosis_))
            print(f"A l'itération {self.nombreIterations_} mu = {round(self.mu_history_[-1],3)} et lambda = {round(self.lambda_history_[-1],3)} et VA = {self.asset_values_[-1]}")
            self.nombreIterations_ += 1

        self.mu_           = self.mu_history_[-1]
        self.lambda_        = self.lambda_history_[-1]

        negInvGaussianDistanceToDefault   = kI(self.asset_values_[-1])
        negInvGaussianDefaultProbability  = 1 if negInvGaussianDistanceToDefault <=0 else defaultProbability(negInvGaussianDistanceToDefault)*100

        return self.nombreIterations_, self.asset_values_[-1], self.mu_, self.lambda_, negInvGaussianDistanceToDefault, negInvGaussianDefaultProbability

    def SymetricVGNaiveMethod(self):

        def parameterNu(kurtosis):
            return kurtosis/3

        def parameterSigma(variance):
            return np.sqrt(variance)

        def omega():
            return (1/self.nu_) * np.log(1-0.5*(self.sigma_**2)*self.nu_)

        def kV(x):
            omega_ = omega()
            return np.log(x/self.companyDebt_) + (self.riskFIR_ + omega_)*self.horizon_

        def modelVG(x):
            return np.exp(-self.riskFIR_ * self.horizon_) * integrate.quad(lambda u: self.varianceGammaPDF(u)*(x*np.exp((self.riskFIR_ + omega()) * self.horizon_ + u) - self.companyDebt_), -kV(x),-10e-10)[0]+np.exp(-self.riskFIR_ * self.horizon_) * integrate.quad(lambda u: self.varianceGammaPDF(u)*(x*np.exp((self.riskFIR_ + omega()) * self.horizon_ + u) - self.companyDebt_), 10e-10,100)[0] - self.equity_value_

        #initialisation des paramètres

        equityRelativeValue = np.diff(np.log(self.companyEquityListValues_), n = 1)
        
        self.sigma_         = parameterSigma(variance(equityRelativeValue))
        self.nu_            = parameterNu(kurtosis(equityRelativeValue))
        
        self.sigma_history_.append(self.sigma_)
        self.nu_history_.append(self.nu_)

        while (np.abs(self.sigma_history_[-1] - self.sigma_history_[-2]) > self.tolerance_) or (np.abs(self.nu_history_[-1] - self.nu_history_[-2]) > self.tolerance_):
            if self.nombreIterations_ >= 12:
                break
            self.asset_values_  = []
            self.sigma_        = self.sigma_history_[-1]
            self.nu_           = self.nu_history_[-1]
            
            for day in range(self.companyEquityListValues_.shape[0]):

                self.relativeTime_ = self.horizon_ + (self.timePeriod_ - day - 1)/self.timePeriod_
                self.equity_value_ = self.companyEquityListValues_[day]
                self.asset_values_.append(optimize.brentq(modelVG,100, 10e13, maxiter = 1000, xtol = 10e-2))

            assetRelativeValue  = np.diff(np.log(self.asset_values_), n = 1)
            kurtosis_           = kurtosis(assetRelativeValue, fisher = False)
            variance_           = variance(assetRelativeValue)*self.timePeriod_
            
            self.sigma_history_.append(parameterSigma(variance_))
            self.nu_history_.append(parameterNu(kurtosis_))
            print(f"A l'itération {self.nombreIterations_} nu = {round(self.nu_history_[-1],3)} et sigma = {round(self.sigma_history_[-1],3)} et VA = {self.asset_values_[-1]}")
            self.nombreIterations_ += 1

        self.nu_           = self.nu_history_[-1]
        self.sigma_        = self.sigma_history_[-1]
        
        varianceGammaNaiveDistanceToDefault     = kV(self.asset_values_[-1])
        varianceGammaNaiveDefaultProbability    = self.varianceGammaCDF(-varianceGammaNaiveDistanceToDefault)

        return self.nombreIterations_, self.asset_values_[-1], self.nu_, self.sigma_, varianceGammaNaiveDistanceToDefault,varianceGammaNaiveDefaultProbability*100

    def SymetricVGModelGLaguerre(self):

        def rootsLaguerre():
            roots, _     = special.roots_laguerre(n, mu=False)
            return roots

        def weightsLaguerre(roots):
            weights = [u_i/((n+1)**2*(special.laguerre(n+1)(u_i))**2) for u_i in roots]
            return weights

        def pFromWeights(roots, weights, i, time):
            num     = weights[i] * (roots[i]**(time - 1))
            den     = np.sum([weight_i * root_i**(time - 1) for (weight_i, root_i) in zip(weights, roots)])
            return num/den

        def omegaBarre(roots, weights):
            sumPwithExp = np.sum([pFromWeights(roots, weights, i, self.relativeTime_/self.nu_) * np.exp(((self.sigma_**2)/2)*self.nu_*roots[i]) for i in range(n)])
            return -np.log(sumPwithExp)/self.relativeTime_

        def parameterNu(kurtosis):
            return kurtosis/3

        def parameterSigma(variance):
            return np.sqrt(variance)

        def d1(x, roots, i, omega):
            num = np.log(x/self.companyDebt_) + (self.riskFIR_ + omega)*(self.relativeTime_ - self.horizon_)+(self.sigma_**2)*self.nu_*roots[i]
            den = self.sigma_ * np.sqrt(self.nu_ * roots[i])
            return num/den
        
        def d2(x, roots, i, omega):
            return d1(x, roots, i, omega) - self.sigma_ * np.sqrt(self.nu_ * roots[i])

        def kV(x):
            return np.log(x/self.companyDebt_) + (self.riskFIR_ + (1/self.nu_) * np.log(1-0.5*(self.sigma_**2)*self.nu_))*self.relativeTime_
        
        def modelVG(x):
            roots       = rootsLaguerre()
            weights     = weightsLaguerre(roots)
            omega       = omegaBarre(roots, weights)
            tau         = self.relativeTime_ - self.horizon_

            leftPart    = x * np.sum([np.exp(omegaBarre(roots, weights)*tau + (0.5**self.nu_*roots[i]*self.sigma_**2))*self.cumulativeGaussianDistribution(d1(x,roots,i,omega))*pFromWeights(roots, weights,i, tau/self.nu_) for i in range(n)])
            rightPart   = self.companyDebt_ * np.exp(-self.riskFIR_ * tau) * np.sum([self.cumulativeGaussianDistribution(d2(x,roots,i,omega)) * pFromWeights(roots, weights,i, tau/self.nu_) for i in range(n)])
            
            return leftPart - rightPart - self.equity_value_

        #initialisation des paramètres
        n                   = 6
        equityRelativeValue = np.diff(np.log(self.companyEquityListValues_), n = 1)
        
        self.sigma_         = parameterSigma(variance(equityRelativeValue)*self.timePeriod_)
        self.nu_            = parameterNu(kurtosis(equityRelativeValue, fisher = False))
        
        self.sigma_history_.append(self.sigma_)
        self.nu_history_.append(self.nu_)


        while (np.abs(self.sigma_history_[-1] - self.sigma_history_[-2]) > self.tolerance_) or (np.abs(self.nu_history_[-1] - self.nu_history_[-2]) > self.tolerance_):
            
            self.asset_values_  = []
            self.sigma_        = self.sigma_history_[-1]
            self.nu_           = self.nu_history_[-1]
            
            for day in range(self.companyEquityListValues_.shape[0]):

                self.relativeTime_ = self.horizon_ + (self.timePeriod_ - day - 1)/self.timePeriod_
                self.equity_value_ = self.companyEquityListValues_[day]
                self.asset_values_.append(optimize.newton(modelVG,self.companyDebt_))#,maxiter=400,tol=1e-6))
            
            assetRelativeValue  = np.diff(np.log(self.asset_values_), n = 1)
            kurtosis_           = kurtosis(assetRelativeValue, fisher = False)
            variance_           = variance(assetRelativeValue)*self.timePeriod_
            
            self.sigma_history_.append(parameterSigma(variance_))
            self.nu_history_.append(parameterNu(kurtosis_))
            print(f"A l'itération {self.nombreIterations_} nu = {round(self.nu_history_[-1],3)} et sigma = {round(self.sigma_history_[-1],3)} et VA = {self.asset_values_[-1]}")
            self.nombreIterations_ += 1

        self.nu_           = self.nu_history_[-1]
        self.sigma_        = self.sigma_history_[-1]
        
        varianceGammaGLaguerreDistanceToDefault     = kV(self.asset_values_[-1])
        varianceGammaGLaguerreDefaultProbability    = self.varianceGammaCDF(-varianceGammaGLaguerreDistanceToDefault)

        return self.nombreIterations_, self.asset_values_[-1], self.nu_, self.sigma_, varianceGammaGLaguerreDistanceToDefault,varianceGammaGLaguerreDefaultProbability*100

    def SymetricVGLewisLipton(self):

        def parameterNu(kurtosis):
            return kurtosis/3

        def parameterSigma(variance):
            return np.sqrt(variance)

        def omega():
            return 1/self.nu_ * np.log(1 - 0.5 * self.nu_ * (self.sigma_**2))

        def phiLL(x,t):
            i   = 1.j
            phi = (1 + 0.5 * self.nu_ * (self.sigma_ * x)**2)**(-t/self.nu_)
            return np.exp(i * x * omega() * t) * phi

        def kV(x):
            return np.log(x/self.companyDebt_) + (self.riskFIR_ + (1/self.nu_) * np.log(1-0.5*(self.sigma_**2)*self.nu_))*self.relativeTime_
        
        def modelVG(x):
            N           = 200
            borneInf    = 0.00001
            borneSup    = 200
            pas         = (borneSup - borneInf)/(N-1)
            subdivision = [borneInf + pas * i for i in range(N)]
            
            i   = 1.j
            tau = self.relativeTime_ - self.horizon_
            k   = np.log(x/self.companyDebt_) + self.riskFIR_ * tau

            def integrandePi1(xi):
                return np.real(np.exp(i*xi*k)*phiLL(xi-i,self.horizon_)/(xi*i))
          
            integrandePi1_  =   [integrandePi1(i) for i in subdivision]
            
            def integrandePi2(xi):
                return np.real(np.exp(i*xi*k)*phiLL(xi,self.horizon_)/(xi*i))

            integrandePi2_  =   [integrandePi2(i) for i in subdivision]

            pi1    = 0.5 + 1/np.pi * pas/2 * (integrandePi1_[0] + 2*np.sum(integrandePi1_[1:-1]) + integrandePi1_[-1])
            pi2    = 0.5 + 1/np.pi * pas/2 * (integrandePi2_[0] + 2*np.sum(integrandePi2_[1:-1]) + integrandePi2_[-1])
            return x*pi1 - self.companyDebt_ * np.exp(-self.riskFIR_ * self.horizon_)*pi2 - self.equity_value_

        #initialisation des paramètres
        equityRelativeValue = np.diff(np.log(self.companyEquityListValues_), n = 1)
        
        self.sigma_         = parameterSigma(variance(equityRelativeValue)*self.timePeriod_)
        self.nu_            = parameterNu(kurtosis(equityRelativeValue, fisher = False))
        
        self.sigma_history_.append(self.sigma_)
        self.nu_history_.append(self.nu_)

        while (np.abs(self.sigma_history_[-1] - self.sigma_history_[-2]) > self.tolerance_) or (np.abs(self.nu_history_[-1] - self.nu_history_[-2]) > self.tolerance_):
            
            self.asset_values_  = []
            self.sigma_        = self.sigma_history_[-1]
            self.nu_           = self.nu_history_[-1]
            
            for day in range(self.companyEquityListValues_.shape[0]):

                self.relativeTime_ = self.horizon_ + (self.timePeriod_ - day - 1)/self.timePeriod_
                self.equity_value_ = self.companyEquityListValues_[day]
                self.asset_values_.append(optimize.newton(modelVG,self.companyDebt_))#,maxiter=400,tol=1e-6))
            
            assetRelativeValue  = np.diff(np.log(self.asset_values_), n = 1)
            kurtosis_           = kurtosis(assetRelativeValue, fisher = False)
            variance_           = variance(assetRelativeValue)*self.timePeriod_
            
            self.sigma_history_.append(parameterSigma(variance_))
            self.nu_history_.append(parameterNu(kurtosis_))
            print(f"A l'itération {self.nombreIterations_} nu = {round(self.nu_history_[-1],3)} et sigma = {round(self.sigma_history_[-1],3)} et VA = {self.asset_values_[-1]}")
            self.nombreIterations_ += 1

        self.nu_           = self.nu_history_[-1]
        self.sigma_        = self.sigma_history_[-1]
        
        varianceGammaLewisLiptonDistanceToDefault   = kV(self.asset_values_[-1])
        varianceGammaLewisLiptonDefaultProbability  = self.varianceGammaCDF(-varianceGammaLewisLiptonDistanceToDefault)

        return self.nombreIterations_, self.asset_values_[-1], self.nu_, self.sigma_, varianceGammaLewisLiptonDistanceToDefault,varianceGammaLewisLiptonDefaultProbability*100

    def SymetricVGCarrMadan(self):

        def parameterNu(kurtosis):
            return kurtosis/3

        def parameterSigma(variance):
            return np.sqrt(variance)

        def omega():
            return 1/self.nu_ * np.log(1 - 0.5 * self.nu_ * (self.sigma_**2))

        
        def kV(x):
            return np.log(x/self.companyDebt_) + (self.riskFIR_ + (1/self.nu_) * np.log(1-0.5*(self.sigma_**2)*self.nu_))*self.relativeTime_
        
        def modelVG(x):
            N           = 500
            borneInf    = 0
            borneSup    = 50
            pas         = (borneSup - borneInf)/(N-1)
            subdivision = [borneInf + pas * i for i in range(N)]

            a           = (np.sqrt(2/self.nu_)/self.sigma_ - 1)/10
            i           = 1j
            tau         = self.relativeTime_ - self.horizon_

            def phiCM(u,t):
                phi = (1 + 0.5 * self.nu_ * (self.sigma_ * u)**2)**(-t/self.nu_)
                return np.exp(i * u * (np.log(x) + (self.riskFIR_ + omega()) * self.horizon_)) * phi

            def integrandeCarrMadan(u):
                return np.real(np.exp(-i*u*np.log(self.companyDebt_)) * phiCM(u-(a+1)*i, self.horizon_)/(a**2 + a - u**2 + i*(2*a+1)*u))

            return np.exp(-a*np.log(self.companyDebt_) - self.riskFIR_*tau)/np.pi * integrate.quad(lambda u: integrandeCarrMadan(u), 0, 50)[0] - self.equity_value_


        #initialisation des paramètres
        equityRelativeValue = np.diff(np.log(self.companyEquityListValues_), n = 1)
        
        self.sigma_         = parameterSigma(variance(equityRelativeValue)*self.timePeriod_)
        self.nu_            = parameterNu(kurtosis(equityRelativeValue, fisher = False))
        
        self.sigma_history_.append(self.sigma_)
        self.nu_history_.append(self.nu_)

        while (np.abs(self.sigma_history_[-1] - self.sigma_history_[-2]) > self.tolerance_) or (np.abs(self.nu_history_[-1] - self.nu_history_[-2]) > self.tolerance_):
            
            self.asset_values_  = []
            self.sigma_        = self.sigma_history_[-1]
            self.nu_           = self.nu_history_[-1]
            
            for day in range(self.companyEquityListValues_.shape[0]):

                self.relativeTime_ = self.horizon_ + (self.timePeriod_ - day - 1)/self.timePeriod_
                self.equity_value_ = self.companyEquityListValues_[day]
                self.asset_values_.append(optimize.newton(modelVG,self.companyDebt_))#,maxiter=400,tol=1e-6))

            assetRelativeValue  = np.diff(np.log(self.asset_values_), n = 1)
            kurtosis_           = kurtosis(assetRelativeValue, fisher = False)
            variance_           = variance(assetRelativeValue)*self.timePeriod_
            
            self.sigma_history_.append(parameterSigma(variance_))
            self.nu_history_.append(parameterNu(kurtosis_))
            print(f"A l'itération {self.nombreIterations_} nu = {round(self.nu_history_[-1],3)} et sigma = {round(self.sigma_history_[-1],3)} et VA = {self.asset_values_[-1]}")
            self.nombreIterations_ += 1

        self.nu_           = self.nu_history_[-1]
        self.sigma_        = self.sigma_history_[-1]
        
        varianceGammaCarrMadanDistanceToDefault         = kV(self.asset_values_[-1])
        varianceGammaLewisCarrMadanDefaultProbability   = self.varianceGammaCDF(-varianceGammaCarrMadanDistanceToDefault)

        return self.nombreIterations_, self.asset_values_[-1], self.nu_, self.sigma_, varianceGammaCarrMadanDistanceToDefault,varianceGammaLewisCarrMadanDefaultProbability*100

    def asymetricVGNaive(self):

        def parametresSigmaThetaNu(parametres):

            (sigma, theta, nu) = parametres

            def momentOrdre1():
                return self.riskFIR_ + 1/nu * np.log(1 - theta * nu - 0.5 * sigma**2 * nu) + theta

            def momentOrdre2():
                return theta**2 * nu + sigma**2

            def momentOrdre3():
                num = theta * nu * (2 * theta**2 * nu + 3 * sigma**2)
                den = (theta**2 * nu + sigma**2)**(3/2)
                return num/den

            def momentOrdre4():
                num = 3 * sigma**4 * nu + 12 * (sigma*nu*theta)**2 + 6 * theta**4 * nu**3 + 3 * sigma**4 + 6 * (sigma*theta)**2 * nu + 3 * theta**4 * nu**2
                den = (theta**2 * nu + sigma**2)**2
                return num/den

            return np.array([momentOrdre2() - self.variance_, momentOrdre3() - self.skewness_, momentOrdre4() - self.kurtosis_])
        
        def omega():
            return 1/self.nu_ * np.log(1 - self.theta_ * self.nu_ - 0.5 * self.nu_ * (self.sigma_**2))
       
        def kV(x):
            return np.log(x/self.companyDebt_) + (self.riskFIR_ + omega()) * self.relativeTime_
        
        def m(p):
            return -1/self.nu_ * np.log(1 - self.theta_ * self.nu_ * p - 0.5 * self.nu_ * (self.sigma_**2) * p**2)
        
        def esscherEquation(p):
            return m(p+1) - m(p)

        def riskNeutralSigma(pStar):
            A = 1 - self.theta_ * self.nu_ * pStar - 0.5 * self.nu_ * (self.sigma_**2) * pStar**2
            return self.sigma_/np.sqrt(A)

        def riskNeutralTheta(pStar):
            A = 1 - self.theta_ * self.nu_ * pStar - 0.5 * self.nu_ * (self.sigma_**2) * pStar**2
            return (self.theta_ + self.sigma_**2 * pStar)/A
        
        def riskNeutralNu():
            return self.nu_
        
        def modelVG_naive(x):
            if self.riskNeutral_:
                return np.exp(-self.riskFIR_ * self.horizon_) * integrate.quad(lambda u: self.varianceGammaPDF(u)*(x*np.exp(u) - self.companyDebt_), -np.log(x/self.companyDebt_),-10e-10)[0]+np.exp(-self.riskFIR_ * self.horizon_) * integrate.quad(lambda u: self.varianceGammaPDF(u)*(x*np.exp(u) - self.companyDebt_), 10e-10,100)[0] - self.equity_value_
            else:
                return np.exp(-self.riskFIR_ * self.horizon_) * integrate.quad(lambda u: self.varianceGammaPDF(u)*(x*np.exp(u + (self.riskFIR_ + omega()) * self.horizon_) - self.companyDebt_), -kV(x),-10e-10)[0]+np.exp(-self.riskFIR_ * self.horizon_) * integrate.quad(lambda u: self.varianceGammaPDF(u)*(x*np.exp(u + (self.riskFIR_ + omega()) * self.horizon_) - self.companyDebt_), 10e-10,100)[0] - self.equity_value_

        #initialisation des paramètres
        equityRelativeValue = np.diff(np.log(self.companyEquityListValues_), n = 1)

        self.mean_          = mean(equityRelativeValue)
        self.variance_      = variance(equityRelativeValue)*self.timePeriod_
        self.kurtosis_      = kurtosis(equityRelativeValue, fisher = False)
        self.skewness_      = skew(equityRelativeValue)

        #Physical parameters
        sigma_, theta_, nu_ = optimize.fsolve(parametresSigmaThetaNu, np.array([1, 0, 1]))

        #Comme fsolve est symétrique par rapport à sigma, on peut se permettre de prendre sigma positif, comme une vol est tjrs positivie
        sigma_              = np.abs(sigma_)

        self.sigma_history_.append(sigma_)
        self.nu_history_.append(nu_)
        self.theta_history_.append(theta_)

        if not self.riskNeutral_: #On reste sur les proba physiques
            while (np.abs(self.sigma_history_[-1] - self.sigma_history_[-2]) > self.tolerance_) or (np.abs(self.nu_history_[-1] - self.nu_history_[-2]) > self.tolerance_) or (np.abs(self.theta_history_[-1] - self.theta_history_[-2]) > self.tolerance_):
                
                self.asset_values_  = []
                self.sigma_         = self.sigma_history_[-1]
                self.nu_            = self.nu_history_[-1]
                self.theta_         = self.theta_history_[-1]
                
                for day in range(self.companyEquityListValues_.shape[0]):

                    self.relativeTime_ = self.horizon_ + (self.timePeriod_ - day - 1)/self.timePeriod_
                    self.equity_value_ = self.companyEquityListValues_[day]
                    self.asset_values_.append(optimize.brentq(modelVG_naive,100, 10e13, maxiter = 1000, xtol = 10e-2))
                
                assetRelativeValue  = np.diff(np.log(self.asset_values_), n = 1)

                self.mean_          = mean(equityRelativeValue)
                self.kurtosis_      = kurtosis(assetRelativeValue, fisher = False)
                self.variance_      = variance(assetRelativeValue)*self.timePeriod_
                self.skewness_      = skew(assetRelativeValue)

                sigma_, theta_, nu_  = optimize.fsolve(parametresSigmaThetaNu, np.array([self.sigma_, self.theta_, self.nu_]))

                self.sigma_history_.append(sigma_)
                self.nu_history_.append(nu_)
                self.theta_history_.append(theta_)

                print(f"A l'itération {self.nombreIterations_} nu = {round(self.nu_history_[-1],3)} et sigma = {round(self.sigma_history_[-1],3)} et theta = {round(self.theta_history_[-1],3)} et VA = {self.asset_values_[-1]}")
                self.nombreIterations_ += 1

            self.nu_        = self.nu_history_[-1]
            self.sigma_     = self.sigma_history_[-1]
            self.theta_     = self.theta_history_[-1]
            
            asymVarianceGammaDistanceToDefault     = kV(self.asset_values_[-1])
            asymVarianceGammaDefaultProbability    = self.varianceGammaCDF(-asymVarianceGammaDistanceToDefault)
            return self.nombreIterations_, self.asset_values_[-1], self.nu_, self.sigma_, self.theta_, asymVarianceGammaDistanceToDefault, asymVarianceGammaDefaultProbability*100

        else: #On calcule les proba risque neutre

            #################################
            ###### Physical Parameters ######
            #################################
            self.sigma_         = self.sigma_history_[-1]
            self.nu_            = self.nu_history_[-1]
            self.theta_         = self.theta_history_[-1]

            pStar   = optimize.fsolve(esscherEquation,0)[0]

            #################################
            #### Risk Neutral Parameters ####
            #################################
            self.sigmaRN_history_.append(riskNeutralSigma(pStar))
            self.thetaRN_history_.append(riskNeutralTheta(pStar))
            self.nuRN_history_.append(riskNeutralNu())

            self.sigmaRN_   = self.sigmaRN_history_[-1]
            self.thetaRN_   = self.thetaRN_history_[-1]
            self.nuRN_      = self.nuRN_history_[-1]

            while (np.abs(self.sigma_history_[-1] - self.sigma_history_[-2]) > self.tolerance_) or (np.abs(self.nu_history_[-1] - self.nu_history_[-2]) > self.tolerance_) or (np.abs(self.theta_history_[-1] - self.theta_history_[-2]) > self.tolerance_):
                
                self.asset_values_  = []

                for day in range(self.companyEquityListValues_.shape[0]):

                    self.relativeTime_ = self.horizon_ + (self.timePeriod_ - day - 1)/self.timePeriod_
                    self.equity_value_ = self.companyEquityListValues_[day]
                    self.asset_values_.append(optimize.brentq(modelVG_naive,100, 10e13, maxiter = 1000, xtol = 10e-2))
                    
                assetRelativeValue  = np.diff(np.log(self.asset_values_), n = 1)

                self.mean_          = mean(equityRelativeValue)
                self.kurtosis_      = kurtosis(assetRelativeValue, fisher = False)
                self.variance_      = variance(assetRelativeValue)*self.timePeriod_
                self.skewness_      = skew(assetRelativeValue)

                #################################
                ###### Physical Parameters ######
                #################################

                sigma_, theta_, nu_  = optimize.fsolve(parametresSigmaThetaNu, np.array([self.sigma_, self.theta_, self.nu_]))
                
                self.sigma_history_.append(sigma_)
                self.nu_history_.append(nu_)
                self.theta_history_.append(theta_)

                self.sigma_         = self.sigma_history_[-1]
                self.nu_            = self.nu_history_[-1]
                self.theta_         = self.theta_history_[-1]
                
                pStar   = optimize.fsolve(esscherEquation, 0)[0]
                
                #################################
                #### Risk Neutral Parameters ####
                #################################

                self.sigmaRN_history_.append(riskNeutralSigma(pStar))
                self.thetaRN_history_.append(riskNeutralTheta(pStar))
                self.nuRN_history_.append(riskNeutralNu())

                self.sigmaRN_       = self.sigmaRN_history_[-1]
                self.thetaRN_       = self.thetaRN_history_[-1]
                self.nuRN_          = self.nuRN_history_[-1]

                print(f"A l'itération {self.nombreIterations_} nu = {round(self.nu_history_[-1],3)} et sigma = {round(self.sigma_history_[-1],3)} et theta = {round(self.theta_history_[-1],3)} et VA = {self.asset_values_[-1]}")
                self.nombreIterations_ += 1            
            
            self.riskNeutral_                       = False
            asymVarianceGammaDistanceToDefault      = - np.log(self.asset_values_[-1]/self.companyDebt_)
            asymVarianceGammaDefaultProbability     = self.varianceGammaCDF(asymVarianceGammaDistanceToDefault)

            return self.nombreIterations_, self.asset_values_[-1], self.nu_, self.sigma_, self.theta_, asymVarianceGammaDistanceToDefault, asymVarianceGammaDefaultProbability*100

    def asymetricVGCarrMadan(self):

        def parametresSigmaThetaNu(parametres):

            (sigma, theta, nu) = parametres

            def momentOrdre1():
                return self.riskFIR_ + 1/nu * np.log(1 - theta * nu - 0.5 * sigma**2 * nu) + theta

            def momentOrdre2():
                return theta**2 * nu + sigma**2

            def momentOrdre3():
                num = theta * nu * (2 * theta**2 * nu + 3 * sigma**2)
                den = (theta**2 * nu + sigma**2)**(3/2)
                return num/den

            def momentOrdre4():
                num = 3 * sigma**4 * nu + 12 * (sigma*nu*theta)**2 + 6 * theta**4 * nu**3 + 3 * sigma**4 + 6 * (sigma*theta)**2 * nu + 3 * theta**4 * nu**2
                den = (theta**2 * nu + sigma**2)**2
                return num/den

            return np.array([momentOrdre2() - self.variance_, momentOrdre3() - self.skewness_, momentOrdre4() - self.kurtosis_])
        
        def omega():
            return 1/self.nu_ * np.log(1 - self.theta_ * self.nu_ - 0.5 * self.nu_ * (self.sigma_**2))
       
        def kV(x):
            return np.log(x/self.companyDebt_) + (self.riskFIR_ + omega())*self.relativeTime_
        
        def m(p):
            return -1/self.nu_ * np.log(1 - self.theta_ * self.nu_ * p - 0.5 * self.nu_ * (self.sigma_**2) * p**2)
        
        def esscherEquation(p):
            return m(p+1) - m(p)

        def riskNeutralSigma(pStar):
            A = 1 - self.theta_ * self.nu_ * pStar - 0.5 * self.nu_ * (self.sigma_**2) * pStar**2
            return self.sigma_/np.sqrt(A)

        def riskNeutralTheta(pStar):
            A = 1 - self.theta_ * self.nu_ * pStar - 0.5 * self.nu_ * (self.sigma_**2) * pStar**2
            return (self.theta_ + self.sigma_**2 * pStar)/A
        
        def riskNeutralNu():
            return self.nu_
        
        def modelVG_CM(x):

            if not self.riskNeutral_:
                a           = (np.sqrt(2/self.nu_)/self.sigma_ - 1)/10
                i           = 1j
                tau         = self.relativeTime_ - self.horizon_

                def phiCM(u,t):
                    PHI = (1 - i * self.theta_ * self.nu_ * u + 0.5 * self.nu_ * (self.sigma_ * u)**2)**(-t/self.nu_)
                    return np.exp(i * u * (np.log(x) + (self.riskFIR_ + omega()) * self.horizon_)) * PHI

                def integrandeCarrMadan(u):
                    return np.real(np.exp(-i*u*np.log(self.companyDebt_)) * phiCM(u-(a+1)*i, self.horizon_)/(a**2 + a - u**2 + i*(2*a+1)*u))

                return np.exp(-a*np.log(self.companyDebt_) - self.riskFIR_*tau)/np.pi * integrate.quad(lambda u: integrandeCarrMadan(u), 0, 50)[0] - self.equity_value_

            else:
                a           = (np.sqrt(2/self.nuRN_)/self.sigmaRN_ - 1)/10
                i           = 1j
                tau         = self.relativeTime_ - self.horizon_

                def psiCM(u,t):
                    PSI = (1 - i * self.thetaRN_ * self.nuRN_ * u + 0.5 * self.nuRN_ * (self.sigmaRN_ * u)**2)**(-t/self.nuRN_)
                    return np.exp(i * u * (np.log(x))) * PSI

                def integrandeCarrMadan(u):
                    return np.real(np.exp(-i*u*np.log(self.companyDebt_)) * psiCM(u-(a+1)*i, self.horizon_)/(a**2 + a - u**2 + i*(2*a+1)*u))
            
                return np.exp(-a*np.log(self.companyDebt_) - self.riskFIR_*tau)/np.pi * integrate.quad(lambda u: integrandeCarrMadan(u), 0, 50)[0] - self.equity_value_

        #initialisation des paramètres
        equityRelativeValue = np.diff(np.log(self.companyEquityListValues_), n = 1)

        self.mean_          = mean(equityRelativeValue)
        self.variance_      = variance(equityRelativeValue)*self.timePeriod_
        self.kurtosis_      = kurtosis(equityRelativeValue, fisher = False)
        self.skewness_      = skew(equityRelativeValue)

        #Physical parameters
        sigma_, theta_, nu_ = optimize.fsolve(parametresSigmaThetaNu, np.array([1, 0, 1]))

        sigma_              = np.abs(sigma_)

        self.sigma_history_.append(sigma_)
        self.nu_history_.append(nu_)
        self.theta_history_.append(theta_)

        if not self.riskNeutral_: 
            while (np.abs(self.sigma_history_[-1] - self.sigma_history_[-2]) > self.tolerance_) or (np.abs(self.nu_history_[-1] - self.nu_history_[-2]) > self.tolerance_) or (np.abs(self.theta_history_[-1] - self.theta_history_[-2]) > self.tolerance_):
                
                self.asset_values_  = []
                self.sigma_         = self.sigma_history_[-1]
                self.nu_            = self.nu_history_[-1]
                self.theta_         = self.theta_history_[-1]
                
                for day in range(self.companyEquityListValues_.shape[0]):

                    self.relativeTime_ = self.horizon_ + (self.timePeriod_ - day - 1)/self.timePeriod_
                    self.equity_value_ = self.companyEquityListValues_[day]
                    self.asset_values_.append(optimize.newton(modelVG_CM,self.companyDebt_))
                
                assetRelativeValue  = np.diff(np.log(self.asset_values_), n = 1)

                self.mean_          = mean(equityRelativeValue)
                self.kurtosis_      = kurtosis(assetRelativeValue, fisher = False)
                self.variance_      = variance(assetRelativeValue)*self.timePeriod_
                self.skewness_      = skew(assetRelativeValue)

                sigma_, theta_, nu_  = optimize.fsolve(parametresSigmaThetaNu, np.array([self.sigma_, self.theta_, self.nu_]))

                self.sigma_history_.append(sigma_)
                self.nu_history_.append(nu_)
                self.theta_history_.append(theta_)

                print(f"A l'itération {self.nombreIterations_} nu = {round(self.nu_history_[-1],3)} et sigma = {round(self.sigma_history_[-1],3)} et theta = {round(self.theta_history_[-1],3)} et VA = {self.asset_values_[-1]}")
                self.nombreIterations_ += 1

            self.nu_        = self.nu_history_[-1]
            self.sigma_     = self.sigma_history_[-1]
            self.theta_     = self.theta_history_[-1]
            
            asymVarianceGammaDistanceToDefault     = kV(self.asset_values_[-1])
            asymVarianceGammaDefaultProbability    = self.varianceGammaCDF(-asymVarianceGammaDistanceToDefault)
            return self.nombreIterations_, self.asset_values_[-1], self.nu_, self.sigma_, self.theta_, asymVarianceGammaDistanceToDefault, asymVarianceGammaDefaultProbability*100

        else:
            #################################
            ###### Physical Parameters ######
            #################################
            self.sigma_         = self.sigma_history_[-1]
            self.nu_            = self.nu_history_[-1]
            self.theta_         = self.theta_history_[-1]

            pStar   = optimize.fsolve(esscherEquation,0)[0]

            #################################
            #### Risk Neutral Parameters ####
            #################################
            self.sigmaRN_history_.append(riskNeutralSigma(pStar))
            self.thetaRN_history_.append(riskNeutralTheta(pStar))
            self.nuRN_history_.append(riskNeutralNu())

            self.sigmaRN_   = self.sigmaRN_history_[-1]
            self.thetaRN_   = self.thetaRN_history_[-1]
            self.nuRN_      = self.nuRN_history_[-1]

            while (np.abs(self.sigma_history_[-1] - self.sigma_history_[-2]) > self.tolerance_) or (np.abs(self.nu_history_[-1] - self.nu_history_[-2]) > self.tolerance_) or (np.abs(self.theta_history_[-1] - self.theta_history_[-2]) > self.tolerance_):
                
                self.asset_values_  = []

                for day in range(self.companyEquityListValues_.shape[0]):

                    self.relativeTime_ = self.horizon_ + (self.timePeriod_ - day - 1)/self.timePeriod_
                    self.equity_value_ = self.companyEquityListValues_[day]
                    self.asset_values_.append(optimize.newton(modelVG_CM,self.companyDebt_))
                
                assetRelativeValue  = np.diff(np.log(self.asset_values_), n = 1)

                self.mean_          = mean(equityRelativeValue)
                self.kurtosis_      = kurtosis(assetRelativeValue, fisher = False)
                self.variance_      = variance(assetRelativeValue)*self.timePeriod_
                self.skewness_      = skew(assetRelativeValue)

                #################################
                ###### Physical Parameters ######
                #################################

                sigma_, theta_, nu_  = optimize.fsolve(parametresSigmaThetaNu, np.array([self.sigma_, self.theta_, self.nu_]))
                
                self.sigma_history_.append(sigma_)
                self.nu_history_.append(nu_)
                self.theta_history_.append(theta_)

                self.sigma_         = self.sigma_history_[-1]
                self.nu_            = self.nu_history_[-1]
                self.theta_         = self.theta_history_[-1]
                
                pStar   = optimize.fsolve(esscherEquation, 0)[0]
                
                #################################
                #### Risk Neutral Parameters ####
                #################################
                self.sigmaRN_history_.append(riskNeutralSigma(pStar))
                self.thetaRN_history_.append(riskNeutralTheta(pStar))
                self.nuRN_history_.append(riskNeutralNu())

                self.sigmaRN_       = self.sigmaRN_history_[-1]
                self.thetaRN_       = self.thetaRN_history_[-1]
                self.nuRN_          = self.nuRN_history_[-1]

                print(f"A l'itération {self.nombreIterations_} nu = {round(self.nu_history_[-1],3)} et sigma = {round(self.sigma_history_[-1],3)} et theta = {round(self.theta_history_[-1],3)} et VA = {self.asset_values_[-1]}")
                self.nombreIterations_ += 1            
            
            self.riskNeutral_                       = False
            asymVarianceGammaDistanceToDefault      = - np.log(self.asset_values_[-1]/self.companyDebt_)
            asymVarianceGammaDefaultProbability     = self.varianceGammaCDF(asymVarianceGammaDistanceToDefault)

            return self.nombreIterations_, self.asset_values_[-1], self.nu_, self.sigma_, self.theta_, asymVarianceGammaDistanceToDefault, asymVarianceGammaDefaultProbability*100

    def bilateralGammaNaive(self):

        def parametresAlphaLambda(parametres):

            (alphaP, alphaM, lambdaP, lambdaM) = parametres

            def momentOrdre1():
                return alphaP/lambdaP - alphaM/lambdaM

            def momentOrdre2():
                return alphaP/(lambdaP)**2 + alphaM/(lambdaM)**2

            def momentOrdre3():
                num = 2 * (alphaP/(lambdaP)**3 - alphaM/(lambdaM)**3)
                den = (alphaP/(lambdaP)**2 + alphaM/(lambdaM)**2)**3/2
                return num/den

            def momentOrdre4():
                num = 6 * (alphaP/(lambdaP)**4 + alphaM/(lambdaM)**4)
                den = (alphaP/(lambdaP)**2 + alphaM/(lambdaM)**2)**2
                return num/den

            return np.array([momentOrdre1() - self.mean_, momentOrdre2() - self.variance_, momentOrdre3() - self.skewness_, momentOrdre4() - self.kurtosis_])
        
        def zeta():
            lP   = self.lambdaP_
            lM   = self.lambdaM_
            aP   = self.alphaP_
            aM   = self.alphaM_

            zeta = -np.log(((lP/(lP - 1))**aP) * (lM/(lM + 1))**aM)

            return zeta

        def kV(x):
            return np.log(x/self.companyDebt_) + (self.riskFIR_ + zeta()) * self.relativeTime_

        def m(p):
            return np.log(((self.lambdaP_/(self.lambdaP_ - p))**self.alphaP_) * (self.lambdaM_/(self.lambdaM_ + p))**self.alphaM_)
        
        def esscherEquation(p):
            return m(p+1) - m(p)

        def riskNeutralParameters(pStar):
            return self.alphaP_, self.alphaM_, self.lambdaP_ - pStar, self.lambdaM_ + pStar

        def modelBilateral(x):
            
            if not riskNeutral:
                return np.exp(-self.riskFIR_ * self.horizon_) * (integrate.quad(lambda u: self.bilateralGammaPDF(u) * (x * np.exp((self.riskFIR_ + zeta()) * self.relativeTime_ + u) - self.companyDebt_), -kV(x), -10e-30)[0] + integrate.quad(lambda u: self.bilateralGammaPDF(u) * (x * np.exp((self.riskFIR_ + zeta()) * self.horizon_ + u) - self.companyDebt_), 10e-30, 3)[0]) - self.equity_value_
            
            else:
                return np.exp(-self.riskFIR_ * self.horizon_) * (integrate.quad(lambda u: self.bilateralGammaPDF(u) * (x * np.exp(u) - self.companyDebt_), -np.log(x/self.companyDebt_), -10e-30)[0] + integrate.quad(lambda u: self.bilateralGammaPDF(u) * (x * np.exp(u) - self.companyDebt_), 10e-30,3)[0]) - self.equity_value_
        

        #initialisation des paramètres
        equityRelativeValue = np.diff(np.log(self.companyEquityListValues_), n = 1)

        self.mean_          = mean(equityRelativeValue)
        self.variance_      = variance(equityRelativeValue)*self.timePeriod_
        self.kurtosis_      = kurtosis(equityRelativeValue, fisher = False)
        self.skewness_      = skew(equityRelativeValue)

        #Physical parameters
        alphaP_, alphaM_, lambdaP_, lambdaM_ = optimize.fsolve(parametresAlphaLambda, np.array([1, 1, 2, 2]))

        self.alphaP_history_.append(alphaP_)
        self.alphaM_history_.append(alphaM_)
        self.lambdaP_history_.append(lambdaP_)
        self.lambdaM_history_.append(lambdaM_)

        if not self.riskNeutral_: #On reste sur les proba physiques
            while (np.abs(self.alphaP_history_[-1] - self.alphaP_history_[-2]) > self.tolerance_) or (np.abs(self.alphaM_history_[-1] - self.alphaM_history_[-2]) > self.tolerance_) or (np.abs(self.lambdaP_history_[-1] - self.lambdaP_history_[-2]) > self.tolerance_) or (np.abs(self.lambdaM_history_[-1] - self.lambdaM_history_[-2]) > self.tolerance_):
                
                self.asset_values_  = []

                self.alphaP_        = self.alphaP_history_[-1]
                self.alphaM_        = self.alphaM_history_[-1]
                self.lambdaP_       = self.lambdaP_history_[-1]
                self.lambdaM_       = self.lambdaM_history_[-1]

                for day in range(self.companyEquityListValues_.shape[0]):
                    print(day)
                    self.relativeTime_ = self.horizon_ + (self.timePeriod_ - day - 1)/self.timePeriod_
                    self.equity_value_ = self.companyEquityListValues_[day]
                    self.asset_values_.append(optimize.brentq(modelBilateral, 100, 10**10, maxiter = 1000, xtol = 10e-2))
                
                assetRelativeValue  = np.diff(np.log(self.asset_values_), n = 1)

                self.mean_          = mean(assetRelativeValue)
                self.kurtosis_      = kurtosis(assetRelativeValue, fisher = False)
                self.variance_      = variance(assetRelativeValue) * self.timePeriod_
                self.skewness_      = skew(assetRelativeValue)

                alphaP_, alphaM_, lambdaP_, lambdaM_ = optimize.fsolve(parametresAlphaLambda, np.array([self.alphaP_, self.alphaM_, self.lambdaP_, self.lambdaM_]))

                self.alphaP_history_.append(alphaP_)
                self.alphaM_history_.append(alphaM_)
                self.lambdaP_history_.append(lambdaP_)
                self.lambdaM_history_.append(lambdaM_)

                print(f"A l'itération {self.nombreIterations_} alpha+ = {round(self.alphaP_history_[-1],3)} et alpha- = {round(self.alphaM_history_[-1],3)} et lambda+ = {round(self.lambdaP_history_[-1],3)} et lambda- = {round(self.lambdaM_history_[-1],3)} et VA = {round(self.asset_values_[-1],2)}")
                self.nombreIterations_ += 1

            self.alphaP_        = self.alphaP_history_[-1]
            self.alphaM_        = self.alphaM_history_[-1]
            self.lambdaP_       = self.lambdaP_history_[-1]
            self.lambdaM_       = self.lambdaM_history_[-1]
            
            bilateralGammaDistanceToDefault     = kV(self.asset_values_[-1])
            bilateralGammaDefaultProbability    = self.bilateralGammaCDF(-bilateralGammaDistanceToDefault)

            return self.nombreIterations_, self.asset_values_[-1], self.alphaP_, self.alphaM_, self.lambdaP_, self.lambdaM_, bilateralGammaDistanceToDefault, bilateralGammaDefaultProbability*100

        else: #On calcule les proba risque neutre

            #################################
            ###### Physical Parameters ######
            #################################
            self.alphaP_        = self.alphaP_history_[-1]
            self.alphaM_        = self.alphaM_history_[-1]
            self.lambdaP_       = self.lambdaP_history_[-1]
            self.lambdaM_       = self.lambdaM_history_[-1]

            pStar   = optimize.fsolve(esscherEquation,0)[0]

            #################################
            #### Risk Neutral Parameters ####
            #################################
            self.alphaP_RN_, self.alphaM_RN_, self.lambdaP_RN_, self.lambdaM_RN_ = riskNeutralParameters(pStar)

            self.alphaP_RN_history_.append(self.alphaP_RN_)
            self.alphaM_RN_history_.append(self.alphaM_RN_)
            self.lambdaP_RN_history_.append(self.lambdaP_RN_)
            self.lambdaM_RN_history_.append(self.lambdaM_RN_)  

            while (np.abs(self.alphaP_history_[-1] - self.alphaP_history_[-2]) > self.tolerance_) or (np.abs(self.alphaM_history_[-1] - self.alphaM_history_[-2]) > self.tolerance_) or (np.abs(self.lambdaP_history_[-1] - self.lambdaP_history_[-2]) > self.tolerance_) or (np.abs(self.lambdaM_history_[-1] - self.lambdaM_history_[-2]) > self.tolerance_):
                
                self.asset_values_  = []

                for day in range(self.companyEquityListValues_.shape[0]):
                    print(day)
                    self.relativeTime_ = self.horizon_ + (self.timePeriod_ - day - 1)/self.timePeriod_
                    self.equity_value_ = self.companyEquityListValues_[day]
                    self.asset_values_.append(optimize.brentq(modelBilateral,100, 300000, maxiter = 1000, xtol = 10e-2))
                    
                assetRelativeValue  = np.diff(np.log(self.asset_values_), n = 1)

                self.mean_          = mean(assetRelativeValue)
                self.kurtosis_      = kurtosis(assetRelativeValue, fisher = False)
                self.variance_      = variance(assetRelativeValue) * self.timePeriod_
                self.skewness_      = skew(assetRelativeValue)

                #################################
                ###### Physical Parameters ######
                #################################

                alphaP_, alphaM_, lambdaP_, lambdaM_ = optimize.fsolve(parametresAlphaLambda, np.array([self.alphaP_, self.alphaM_, self.lambdaP_, self.lambdaM_]))                
                
                self.alphaP_history_.append(alphaP_)
                self.alphaM_history_.append(alphaM_)
                self.lambdaP_history_.append(lambdaP_)
                self.lambdaM_history_.append(lambdaM_)

                self.alphaP_    = self.alphaP_history_[-1]
                self.alphaM_    = self.alphaM_history_[-1]
                self.lambdaP_   = self.lambdaP_history_[-1]
                self.lambdaM_   = self.lambdaM_history_[-1]
                
                pStar   = optimize.fsolve(esscherEquation, 0)[0]
                
                #################################
                #### Risk Neutral Parameters ####
                #################################

                self.alphaP_RN_, self.alphaM_RN_, self.lambdaP_RN_, self.lambdaM_RN_ = riskNeutralParameters(pStar)

                self.alphaP_RN_history_.append(self.alphaP_RN_)
                self.alphaM_RN_history_.append(self.alphaM_RN_)
                self.lambdaP_RN_history_.append(self.lambdaP_RN_)
                self.lambdaM_RN_history_.append(self.lambdaM_RN_)  

                print(f"A l'itération {self.nombreIterations_} alpha+ = {round(self.alphaP_history_[-1],3)} et alpha- = {round(self.alphaM_history_[-1],3)} et lambda+ = {round(self.lambdaP_history_[-1],3)} et lambda- = {round(self.lambdaM_history_[-1],3)} et VA = {round(self.asset_values_[-1],2)}")
                self.nombreIterations_ += 1            
            
            self.riskNeutral_                    = False
            bilateralGammaDistanceToDefault      = - np.log(self.asset_values_[-1]/self.companyDebt_)
            bilateralGammaDefaultProbability     = self.bilateralGammaCDF(bilateralGammaDistanceToDefault)

            return self.nombreIterations_, self.asset_values_[-1], self.alphaP_, self.alphaM_, self.lambdaP_, self.lambdaM_, bilateralGammaDistanceToDefault, bilateralGammaDefaultProbability*100

    def bilateralGammaCarrMadan(self):

        def parametresAlphaLambda(parametres):

            (alphaP, alphaM, lambdaP, lambdaM) = parametres

            def momentOrdre1():
                return alphaP/lambdaP - alphaM/lambdaM

            def momentOrdre2():
                return alphaP/(lambdaP)**2 + alphaM/(lambdaM)**2

            def momentOrdre3():
                num = 2 * (alphaP/(lambdaP)**3 - alphaM/(lambdaM)**3)
                den = (alphaP/(lambdaP)**2 + alphaM/(lambdaM)**2)**3/2
                return num/den

            def momentOrdre4():
                num = 6 * (alphaP/(lambdaP)**4 + alphaM/(lambdaM)**4)
                den = (alphaP/(lambdaP)**2 + alphaM/(lambdaM)**2)**2
                return num/den

            return np.array([momentOrdre1() - self.mean_, momentOrdre2() - self.variance_, momentOrdre3() - self.skewness_, momentOrdre4() - self.kurtosis_])
        
        def zeta():
            lP   = self.lambdaP_
            lM   = self.lambdaM_
            aP   = self.alphaP_
            aM   = self.alphaM_

            zeta = -np.log(((lP/(lP - 1))**aP) * (lM/(lM + 1))**aM)

            return zeta

        def kV(x):
            return np.log(x/self.companyDebt_) + (self.riskFIR_ + zeta()) * self.relativeTime_

        def m(p):
            return np.log(((self.lambdaP_/(self.lambdaP_ - p))**self.alphaP_) * (self.lambdaM_/(self.lambdaM_ + p))**self.alphaM_)
        
        def esscherEquation(p):
            return m(p+1) - m(p)

        def riskNeutralParameters(pStar):
            return self.alphaP_, self.alphaM_, self.lambdaP_ - pStar, self.lambdaM_ + pStar

        def modelBilateralCM(x):

            if not self.riskNeutral_:
                a           = 0.1#(np.sqrt(2/self.nu_)/self.sigma_ - 1)/10 #????
                i           = 1j
                tau         = self.relativeTime_ - self.horizon_

                def psiCM(u,t):
                    PSI = ((self.lambdaP_/(self.lambdaP_ - i * u))**(self.alphaP_*t)) * (self.lambdaM_/(self.lambdaM_ + i * u))**(self.alphaM_ * t)
                    return np.exp(i * u * (np.log(x) + (self.riskFIR_ + zeta()) * self.horizon_)) * PSI

                def integrandeCarrMadan(u):
                    return np.real(np.exp(-i*u*np.log(self.companyDebt_)) * psiCM(u-(a+1)*i, self.horizon_)/(a**2 + a - u**2 + i*(2*a+1)*u))

                return np.exp(-a*np.log(self.companyDebt_) - self.riskFIR_*tau)/np.pi * integrate.quad(lambda u: integrandeCarrMadan(u), 0, 50)[0] - self.equity_value_

            else:
                a           = 0.1#(np.sqrt(2/self.nuRN_)/self.sigmaRN_ - 1)/10
                i           = 1j
                tau         = self.relativeTime_ - self.horizon_
                
                def psiCM(u,t):
                    PSI = ((self.lambdaP_RN_/(self.lambdaP_RN_ - i * u))**(self.alphaP_RN_*t)) * (self.lambdaM_RN_/(self.lambdaM_RN_ + i * u))**(self.alphaM_RN_ * t)
                    return np.exp(i * u * np.log(x)) * PSI

                def integrandeCarrMadan(u):
                    return np.real(np.exp(-i*u*np.log(self.companyDebt_)) * psiCM(u-(a+1)*i, self.horizon_)/(a**2 + a - u**2 + i*(2*a+1)*u))
            
                return np.exp(-a*np.log(self.companyDebt_) - self.riskFIR_*tau)/np.pi * integrate.quad(lambda u: integrandeCarrMadan(u), 0, 50)[0] - self.equity_value_

        #initialisation des paramètres
        equityRelativeValue = np.diff(np.log(self.companyEquityListValues_), n = 1)

        self.mean_          = mean(equityRelativeValue)
        self.variance_      = variance(equityRelativeValue)*self.timePeriod_
        self.kurtosis_      = kurtosis(equityRelativeValue, fisher = False)
        self.skewness_      = skew(equityRelativeValue)

        #Physical parameters
        alphaP_, alphaM_, lambdaP_, lambdaM_ = optimize.fsolve(parametresAlphaLambda, np.array([1, 1, 2, 2]))

        self.alphaP_history_.append(alphaP_)
        self.alphaM_history_.append(alphaM_)
        self.lambdaP_history_.append(lambdaP_)
        self.lambdaM_history_.append(lambdaM_)

        if not self.riskNeutral_: #On reste sur les proba physiques
            while (np.abs(self.alphaP_history_[-1] - self.alphaP_history_[-2]) > self.tolerance_) or (np.abs(self.alphaM_history_[-1] - self.alphaM_history_[-2]) > self.tolerance_) or (np.abs(self.lambdaP_history_[-1] - self.lambdaP_history_[-2]) > self.tolerance_) or (np.abs(self.lambdaM_history_[-1] - self.lambdaM_history_[-2]) > self.tolerance_):
                
                self.asset_values_  = []

                self.alphaP_        = self.alphaP_history_[-1]
                self.alphaM_        = self.alphaM_history_[-1]
                self.lambdaP_       = self.lambdaP_history_[-1]
                self.lambdaM_       = self.lambdaM_history_[-1]

                for day in range(self.companyEquityListValues_.shape[0]):
                    self.relativeTime_ = self.horizon_ + (self.timePeriod_ - day - 1)/self.timePeriod_
                    self.equity_value_ = self.companyEquityListValues_[day]
                    self.asset_values_.append(optimize.newton(modelBilateralCM, self.companyDebt_))
                
                assetRelativeValue  = np.diff(np.log(self.asset_values_), n = 1)

                self.mean_          = mean(assetRelativeValue)
                self.kurtosis_      = kurtosis(assetRelativeValue, fisher = False)
                self.variance_      = variance(assetRelativeValue) * self.timePeriod_
                self.skewness_      = skew(assetRelativeValue)

                alphaP_, alphaM_, lambdaP_, lambdaM_ = optimize.fsolve(parametresAlphaLambda, np.array([self.alphaP_, self.alphaM_, self.lambdaP_, self.lambdaM_]))

                self.alphaP_history_.append(alphaP_)
                self.alphaM_history_.append(alphaM_)
                self.lambdaP_history_.append(lambdaP_)
                self.lambdaM_history_.append(lambdaM_)

                print(f"A l'itération {self.nombreIterations_} alpha+ = {round(self.alphaP_history_[-1],3)} et alpha- = {round(self.alphaM_history_[-1],3)} et lambda+ = {round(self.lambdaP_history_[-1],3)} et lambda- = {round(self.lambdaM_history_[-1],3)} et VA = {round(self.asset_values_[-1],2)}")
                self.nombreIterations_ += 1

            self.alphaP_        = self.alphaP_history_[-1]
            self.alphaM_        = self.alphaM_history_[-1]
            self.lambdaP_       = self.lambdaP_history_[-1]
            self.lambdaM_       = self.lambdaM_history_[-1]
            
            bilateralGammaDistanceToDefault     = kV(self.asset_values_[-1])
            bilateralGammaDefaultProbability    = self.bilateralGammaCDF(-bilateralGammaDistanceToDefault)

            return self.nombreIterations_, self.asset_values_[-1], self.alphaP_, self.alphaM_, self.lambdaP_, self.lambdaM_, bilateralGammaDistanceToDefault, bilateralGammaDefaultProbability*100

        else: #On calcule les proba risque neutre

            #################################
            ###### Physical Parameters ######
            #################################
            self.alphaP_        = self.alphaP_history_[-1]
            self.alphaM_        = self.alphaM_history_[-1]
            self.lambdaP_       = self.lambdaP_history_[-1]
            self.lambdaM_       = self.lambdaM_history_[-1]

            pStar   = optimize.fsolve(esscherEquation,0)[0]

            #################################
            #### Risk Neutral Parameters ####
            #################################
            self.alphaP_RN_, self.alphaM_RN_, self.lambdaP_RN_, self.lambdaM_RN_ = riskNeutralParameters(pStar)

            self.alphaP_RN_history_.append(self.alphaP_RN_)
            self.alphaM_RN_history_.append(self.alphaM_RN_)
            self.lambdaP_RN_history_.append(self.lambdaP_RN_)
            self.lambdaM_RN_history_.append(self.lambdaM_RN_)  

            while (np.abs(self.alphaP_history_[-1] - self.alphaP_history_[-2]) > self.tolerance_) or (np.abs(self.alphaM_history_[-1] - self.alphaM_history_[-2]) > self.tolerance_) or (np.abs(self.lambdaP_history_[-1] - self.lambdaP_history_[-2]) > self.tolerance_) or (np.abs(self.lambdaM_history_[-1] - self.lambdaM_history_[-2]) > self.tolerance_):
                
                self.asset_values_  = []

                for day in range(self.companyEquityListValues_.shape[0]):
                    #print(day)
                    self.relativeTime_ = self.horizon_ + (self.timePeriod_ - day - 1)/self.timePeriod_
                    self.equity_value_ = self.companyEquityListValues_[day]
                    self.asset_values_.append(optimize.newton(modelBilateralCM, self.companyDebt_))
                    
                assetRelativeValue  = np.diff(np.log(self.asset_values_), n = 1)

                self.mean_          = mean(assetRelativeValue)
                self.kurtosis_      = kurtosis(assetRelativeValue, fisher = False)
                self.variance_      = variance(assetRelativeValue) * self.timePeriod_
                self.skewness_      = skew(assetRelativeValue)

                #################################
                ###### Physical Parameters ######
                #################################

                alphaP_, alphaM_, lambdaP_, lambdaM_ = optimize.fsolve(parametresAlphaLambda, np.array([self.alphaP_, self.alphaM_, self.lambdaP_, self.lambdaM_]))                
                
                self.alphaP_history_.append(alphaP_)
                self.alphaM_history_.append(alphaM_)
                self.lambdaP_history_.append(lambdaP_)
                self.lambdaM_history_.append(lambdaM_)

                self.alphaP_    = self.alphaP_history_[-1]
                self.alphaM_    = self.alphaM_history_[-1]
                self.lambdaP_   = self.lambdaP_history_[-1]
                self.lambdaM_   = self.lambdaM_history_[-1]
                
                pStar   = optimize.fsolve(esscherEquation, 0)[0]
                
                #################################
                #### Risk Neutral Parameters ####
                #################################

                self.alphaP_RN_, self.alphaM_RN_, self.lambdaP_RN_, self.lambdaM_RN_ = riskNeutralParameters(pStar)

                self.alphaP_RN_history_.append(self.alphaP_RN_)
                self.alphaM_RN_history_.append(self.alphaM_RN_)
                self.lambdaP_RN_history_.append(self.lambdaP_RN_)
                self.lambdaM_RN_history_.append(self.lambdaM_RN_)  

                print(f"A l'itération {self.nombreIterations_} alpha+ = {round(self.alphaP_history_[-1],3)} et alpha- = {round(self.alphaM_history_[-1],3)} et lambda+ = {round(self.lambdaP_history_[-1],3)} et lambda- = {round(self.lambdaM_history_[-1],3)} et VA = {round(self.asset_values_[-1],2)}")
                self.nombreIterations_ += 1            
            
            self.riskNeutral_                    = False
            bilateralGammaDistanceToDefault      = - np.log(self.asset_values_[-1]/self.companyDebt_)
            bilateralGammaDefaultProbability     = self.bilateralGammaCDF(bilateralGammaDistanceToDefault)

            return self.nombreIterations_, self.asset_values_[-1], self.alphaP_, self.alphaM_, self.lambdaP_, self.lambdaM_, bilateralGammaDistanceToDefault, bilateralGammaDefaultProbability*100

    def bilateralGammaLewisLipton(self):

        def parametresAlphaLambda(parametres):

            (alphaP, alphaM, lambdaP, lambdaM) = parametres

            def momentOrdre1():
                return alphaP/lambdaP - alphaM/lambdaM

            def momentOrdre2():
                return alphaP/(lambdaP)**2 + alphaM/(lambdaM)**2

            def momentOrdre3():
                num = 2 * (alphaP/(lambdaP)**3 - alphaM/(lambdaM)**3)
                den = (alphaP/(lambdaP)**2 + alphaM/(lambdaM)**2)**3/2
                return num/den

            def momentOrdre4():
                num = 6 * (alphaP/(lambdaP)**4 + alphaM/(lambdaM)**4)
                den = (alphaP/(lambdaP)**2 + alphaM/(lambdaM)**2)**2
                return num/den

            return np.array([momentOrdre1() - self.mean_, momentOrdre2() - self.variance_, momentOrdre3() - self.skewness_, momentOrdre4() - self.kurtosis_])
        
        def zeta():
            lP   = self.lambdaP_
            lM   = self.lambdaM_
            aP   = self.alphaP_
            aM   = self.alphaM_

            zeta = -np.log(((lP/(lP - 1))**aP) * (lM/(lM + 1))**aM)

            return zeta

        def kV(x):
            return np.log(x/self.companyDebt_) + (self.riskFIR_ + zeta()) * self.relativeTime_

        def PSI(x,t):
            i   = 1.j
            if not self.riskNeutral_:
                return ((self.lambdaP_/(self.lambdaP_ - i * x))**(self.alphaP_*t)) * (self.lambdaM_/(self.lambdaM_ + i * x))**(self.alphaM_ * t)
            else:
                return ((self.lambdaP_RN_/(self.lambdaP_RN_ - i * x))**(self.alphaP_RN_*t)) * (self.lambdaM_RN_/(self.lambdaM_RN_ + i * x))**(self.alphaM_RN_ * t)

        def m(p):
            return np.log(((self.lambdaP_/(self.lambdaP_ - p))**self.alphaP_) * (self.lambdaM_/(self.lambdaM_ + p))**self.alphaM_)
        
        def esscherEquation(p):
            return m(p+1) - m(p)

        def riskNeutralParameters(pStar):
            return self.alphaP_, self.alphaM_, self.lambdaP_ - pStar, self.lambdaM_ + pStar

        def modelBilateralLL(x):

            i   = 1.j
            tau = self.relativeTime_ - self.horizon_
            k   = np.log(x/self.companyDebt_) + self.riskFIR_ * tau

            integralePi1    = integrate.quad(lambda u: np.real(np.exp(i*u*k)*PSI(u-i,self.horizon_)/(u*i)), 0, 50)[0]
            integralePi2    = integrate.quad(lambda u: np.real(np.exp(i*u*k)*PSI(u,self.horizon_)/(u*i)), 0, 50)[0]
            
            return  x * (0.5 + integralePi1/np.pi) - self.companyDebt_ * np.exp(-self.riskFIR_ * self.horizon_) * (0.5 + integralePi2/np.pi) - self.equity_value_

        #initialisation des paramètres
        equityRelativeValue = np.diff(np.log(self.companyEquityListValues_), n = 1)

        self.mean_          = mean(equityRelativeValue)
        self.variance_      = variance(equityRelativeValue)*self.timePeriod_
        self.kurtosis_      = kurtosis(equityRelativeValue, fisher = False)
        self.skewness_      = skew(equityRelativeValue)

        #Physical parameters
        alphaP_, alphaM_, lambdaP_, lambdaM_ = optimize.fsolve(parametresAlphaLambda, np.array([1, 1, 2, 2]))

        self.alphaP_history_.append(alphaP_)
        self.alphaM_history_.append(alphaM_)
        self.lambdaP_history_.append(lambdaP_)
        self.lambdaM_history_.append(lambdaM_)

        if not self.riskNeutral_: #On reste sur les proba physiques
            while (np.abs(self.alphaP_history_[-1] - self.alphaP_history_[-2]) > self.tolerance_) or (np.abs(self.alphaM_history_[-1] - self.alphaM_history_[-2]) > self.tolerance_) or (np.abs(self.lambdaP_history_[-1] - self.lambdaP_history_[-2]) > self.tolerance_) or (np.abs(self.lambdaM_history_[-1] - self.lambdaM_history_[-2]) > self.tolerance_):
                
                self.asset_values_  = []

                self.alphaP_        = self.alphaP_history_[-1]
                self.alphaM_        = self.alphaM_history_[-1]
                self.lambdaP_       = self.lambdaP_history_[-1]
                self.lambdaM_       = self.lambdaM_history_[-1]

                for day in range(self.companyEquityListValues_.shape[0]):
                    #print(day)
                    self.relativeTime_ = self.horizon_ + (self.timePeriod_ - day - 1)/self.timePeriod_
                    self.equity_value_ = self.companyEquityListValues_[day]
                    self.asset_values_.append(optimize.newton(modelBilateralLL, self.companyDebt_))
                
                assetRelativeValue  = np.diff(np.log(self.asset_values_), n = 1)

                self.mean_          = mean(assetRelativeValue)
                self.kurtosis_      = kurtosis(assetRelativeValue, fisher = False)
                self.variance_      = variance(assetRelativeValue) * self.timePeriod_
                self.skewness_      = skew(assetRelativeValue)

                alphaP_, alphaM_, lambdaP_, lambdaM_ = optimize.fsolve(parametresAlphaLambda, np.array([self.alphaP_, self.alphaM_, self.lambdaP_, self.lambdaM_]))

                self.alphaP_history_.append(alphaP_)
                self.alphaM_history_.append(alphaM_)
                self.lambdaP_history_.append(lambdaP_)
                self.lambdaM_history_.append(lambdaM_)

                print(f"A l'itération {self.nombreIterations_} alpha+ = {round(self.alphaP_history_[-1],3)} et alpha- = {round(self.alphaM_history_[-1],3)} et lambda+ = {round(self.lambdaP_history_[-1],3)} et lambda- = {round(self.lambdaM_history_[-1],3)} et VA = {round(self.asset_values_[-1],2)}")
                self.nombreIterations_ += 1

            self.alphaP_        = self.alphaP_history_[-1]
            self.alphaM_        = self.alphaM_history_[-1]
            self.lambdaP_       = self.lambdaP_history_[-1]
            self.lambdaM_       = self.lambdaM_history_[-1]
            
            bilateralGammaDistanceToDefault     = kV(self.asset_values_[-1])
            bilateralGammaDefaultProbability    = self.bilateralGammaCDF(-bilateralGammaDistanceToDefault)

            return self.nombreIterations_, self.asset_values_[-1], self.alphaP_, self.alphaM_, self.lambdaP_, self.lambdaM_, bilateralGammaDistanceToDefault, bilateralGammaDefaultProbability*100

        else: #On calcule les proba risque neutre

            #################################
            ###### Physical Parameters ######
            #################################
            self.alphaP_        = self.alphaP_history_[-1]
            self.alphaM_        = self.alphaM_history_[-1]
            self.lambdaP_       = self.lambdaP_history_[-1]
            self.lambdaM_       = self.lambdaM_history_[-1]

            pStar   = optimize.fsolve(esscherEquation,0)[0]

            #################################
            #### Risk Neutral Parameters ####
            #################################
            self.alphaP_RN_, self.alphaM_RN_, self.lambdaP_RN_, self.lambdaM_RN_ = riskNeutralParameters(pStar)

            self.alphaP_RN_history_.append(self.alphaP_RN_)
            self.alphaM_RN_history_.append(self.alphaM_RN_)
            self.lambdaP_RN_history_.append(self.lambdaP_RN_)
            self.lambdaM_RN_history_.append(self.lambdaM_RN_)  

            while (np.abs(self.alphaP_history_[-1] - self.alphaP_history_[-2]) > self.tolerance_) or (np.abs(self.alphaM_history_[-1] - self.alphaM_history_[-2]) > self.tolerance_) or (np.abs(self.lambdaP_history_[-1] - self.lambdaP_history_[-2]) > self.tolerance_) or (np.abs(self.lambdaM_history_[-1] - self.lambdaM_history_[-2]) > self.tolerance_):
                
                self.asset_values_  = []

                for day in range(self.companyEquityListValues_.shape[0]):
                    #print(day)
                    self.relativeTime_ = self.horizon_ + (self.timePeriod_ - day - 1)/self.timePeriod_
                    self.equity_value_ = self.companyEquityListValues_[day]
                    self.asset_values_.append(optimize.newton(modelBilateralLL, self.companyDebt_))
                    
                assetRelativeValue  = np.diff(np.log(self.asset_values_), n = 1)

                self.mean_          = mean(assetRelativeValue)
                self.kurtosis_      = kurtosis(assetRelativeValue, fisher = False)
                self.variance_      = variance(assetRelativeValue) * self.timePeriod_
                self.skewness_      = skew(assetRelativeValue)

                #################################
                ###### Physical Parameters ######
                #################################

                alphaP_, alphaM_, lambdaP_, lambdaM_ = optimize.fsolve(parametresAlphaLambda, np.array([self.alphaP_, self.alphaM_, self.lambdaP_, self.lambdaM_]))                
                
                self.alphaP_history_.append(alphaP_)
                self.alphaM_history_.append(alphaM_)
                self.lambdaP_history_.append(lambdaP_)
                self.lambdaM_history_.append(lambdaM_)

                self.alphaP_    = self.alphaP_history_[-1]
                self.alphaM_    = self.alphaM_history_[-1]
                self.lambdaP_   = self.lambdaP_history_[-1]
                self.lambdaM_   = self.lambdaM_history_[-1]
                
                pStar   = optimize.fsolve(esscherEquation, 0)[0]
                
                #################################
                #### Risk Neutral Parameters ####
                #################################

                self.alphaP_RN_, self.alphaM_RN_, self.lambdaP_RN_, self.lambdaM_RN_ = riskNeutralParameters(pStar)

                self.alphaP_RN_history_.append(self.alphaP_RN_)
                self.alphaM_RN_history_.append(self.alphaM_RN_)
                self.lambdaP_RN_history_.append(self.lambdaP_RN_)
                self.lambdaM_RN_history_.append(self.lambdaM_RN_)  

                print(f"A l'itération {self.nombreIterations_} alpha+ = {round(self.alphaP_history_[-1],3)} et alpha- = {round(self.alphaM_history_[-1],3)} et lambda+ = {round(self.lambdaP_history_[-1],3)} et lambda- = {round(self.lambdaM_history_[-1],3)} et VA = {round(self.asset_values_[-1],2)}")
                self.nombreIterations_ += 1            
            
            self.riskNeutral_                    = False
            bilateralGammaDistanceToDefault      = - np.log(self.asset_values_[-1]/self.companyDebt_)
            bilateralGammaDefaultProbability     = self.bilateralGammaCDF(bilateralGammaDistanceToDefault)

            return self.nombreIterations_, self.asset_values_[-1], self.alphaP_, self.alphaM_, self.lambdaP_, self.lambdaM_, bilateralGammaDistanceToDefault, bilateralGammaDefaultProbability*100

if __name__ == "__main__":

    print("\n" + "#"*120) 
    print("#"*120 + "\n") 
    model = input("Quel modèle voulez-vous utiliser parmis Merton, NegGamma, NegIG, VG_GL, VG_naive, VG_Lewis, VG_CM, VG_asymCM, VG_asymNaive, bilateral? ")
    print("\n" + "#"*120 ) 
    print("#"*120 + "\n") 

    if model == 'NegGamma':

        #####################################################
        ##################### NEG GAMMA #####################
        #####################################################

        print("\n" + "#"*120 + "\n")
        horizon = input("Entrez l'horizon (en années) auquel on s'intérèsse : ")
        print("\n" + "#"*120 + "\n")   

        valuesNegGamma = pd.DataFrame({'ticker':[], 'VA': [], 'lambda': [], 'rho': [], 'distDefault':[], 'probaDefault':[], 'nIter': [], 'execTime': []})
        tickers = ['SAP GY', 'MRK GY', 'AI FP', 'SU FP', 'CRH LN', 'DAI GY', 'VIE FP', 'SRG IM', 'AMP IM', 'FR FP', 'EO FP', 'GET FP', 'LHA GY', 'PIA IM', 'CO FP']

        for i in range(len(tickers)):

            ticker = tickers[i]

            print("\n" + "#"*120 + "\n")
            print(f"La boîte à laquelle on s'intérèsse est : {ticker}")
            print("\n" + "#"*120 + "\n")

            try:    
                start_time = time.time()

                NegGamma = calibrationModels(companyEquityTicker = ticker + ' Equity', horizon = int(horizon))
                nIter, assetValue, rho, lambd, distToDefault, defaultProba = NegGamma.NegativeGammaModel()

                execTime = round(time.time() - start_time, 4)
                print("Le temps d'execution pour Negative Gamma est de %s secondes" % execTime)
                
                if valuesNegGamma.empty:
                    valuesNegGamma = pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'lambda': [lambd], 'rho': [rho], 'distDefault':[distToDefault], 'probaDefault': [defaultProba], 'nIter': [nIter], 'execTime': [execTime]})
                else:
                    valuesNegGamma = pd.concat([valuesNegGamma, pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'lambda': [lambd], 'rho': [rho], 'distDefault':[distToDefault], 'probaDefault': [defaultProba], 'nIter': [nIter], 'execTime': [execTime]})], ignore_index=True)
            
            except:
                print(f"Pas de données sur {ticker}")
        print(valuesNegGamma)

    elif model =='NegIG':

        #####################################################
        #################### NegIG GAMMA ####################
        #####################################################
        
        print("\n" + "#"*120 + "\n")
        horizon = input("Entrez l'horizon (en années) auquel on s'intérèsse : ")
        print("\n" + "#"*120 + "\n")   

        valuesNegIG = pd.DataFrame({'ticker':[], 'VA': [], 'mu': [], 'lambda': [], 'distDefault':[], 'probaDefault':[], 'nIter': [], 'execTime': []})
        tickers = ['SAP GY', 'MRK GY', 'AI FP', 'SU FP', 'CRH LN', 'DAI GY', 'VIE FP', 'SRG IM', 'AMP IM', 'FR FP', 'EO FP', 'GET FP', 'LHA GY', 'PIA IM', 'CO FP']

        for i in range(len(tickers)):

            ticker = tickers[i]

            print("\n" + "#"*120 + "\n")
            print(f"La boîte à laquelle on s'intérèsse est : {ticker}")
            print("\n" + "#"*120 + "\n")

            try:    
                start_time = time.time()

                NegIG = calibrationModels(companyEquityTicker = ticker + ' Equity', horizon = int(horizon))
                nIter, assetValue, mu, lambd, distToDefault, defaultProba = NegIG.NegInvGaussianModel()

                execTime = round(time.time() - start_time, 4)

                print("Le temps d'execution pour Negative IG est de %s secondes" % execTime)

                if valuesNegIG.empty:
                    valuesNegIG = pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'mu': [mu], 'lmabda': [lambd], 'distDefault':[distToDefault], 'probaDefault': [defaultProba], 'nIter': [nIter], 'execTime': [execTime]})
                else:
                    valuesNegIG = pd.concat([valuesNegIG, pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'mu': [mu], 'lmabda': [lambd], 'distDefault':[distToDefault], 'probaDefault': [defaultProba], 'nIter': [nIter], 'execTime': [execTime]})], ignore_index=True)

            except:
                print(f"Pas de données sur {ticker}")
        print(valuesNegIG)

    elif model == "Merton":

        #####################################################
        ####################### Merton ######################
        #####################################################

        print("\n" + "#"*120 + "\n")
        horizon = input("Entrez l'horizon (en années) auquel on s'intérèsse : ")
        print("\n" + "#"*120 + "\n")   

        valuesMerton = pd.DataFrame({'ticker':[], 'VA': [], 'sigmaA': [], 'distDefault':[], 'probaDefault':[], 'nIter': [], 'execTime': []})
        tickers = ['SAP GY', 'MRK GY', 'AI FP', 'SU FP', 'CRH LN', 'DAI GY', 'VIE FP', 'SRG IM', 'AMP IM', 'FR FP', 'EO FP', 'GET FP', 'LHA GY', 'PIA IM', 'CO FP']

        for i in range(len(tickers)):

            ticker = tickers[i]

            print("\n" + "#"*120 + "\n")
            print(f"La boîte à laquelle on s'intérèsse est : {ticker}")
            print("\n" + "#"*120 + "\n")

            try:    
                start_time = time.time()

                merton = calibrationModels(companyEquityTicker = ticker + ' Equity', horizon = int(horizon))
                nIter, assetValue, sigmaA, distToDefault, defaultProba = merton.BlackScholesMertonModel()

                execTime = round(time.time() - start_time, 4)

                print("Le temps d'execution pour Merton est de %s secondes" % execTime)

                if valuesMerton.empty:
                    valuesMerton = pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'sigmaA': [sigmaA], 'distDefault':[distToDefault], 'probaDefault': [defaultProba], 'nIter': [nIter], 'execTime': [execTime]})
                else:
                    valuesMerton = pd.concat([valuesMerton,pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'sigmaA': [sigmaA], 'distDefault':[distToDefault], 'probaDefault': [defaultProba], 'nIter': [nIter], 'execTime': [execTime]})], ignore_index=True)

            except:
                print(f"Pas de données sur {ticker}")
        print(valuesMerton)

    elif model =='VG_GL':

        #####################################################
        ########## Variance GAMMA Gauss-Laguerre ############
        #####################################################
        
        print("\n" + "#"*120 + "\n")
        horizon = input("Entrez l'horizon (en années) auquel on s'intérèsse : ")
        print("\n" + "#"*120 + "\n")

        valuesVG_GL = pd.DataFrame({'ticker':[], 'VA': [], 'nu': [], 'sigma': [], 'distDefault':[], 'probaDefault':[],'nIter': [], 'execTime': []})
        tickers = ['SAP GY', 'MRK GY', 'AI FP', 'SU FP', 'CRH LN', 'DAI GY', 'VIE FP', 'SRG IM', 'AMP IM', 'FR FP', 'EO FP', 'GET FP', 'LHA GY', 'PIA IM', 'CO FP']

        for i in range(len(tickers)):

            ticker = tickers[i]

            print("\n" + "#"*120 + "\n")
            print(f"La boîte à laquelle on s'intérèsse est : {ticker}")
            print("\n" + "#"*120 + "\n")

            try:    
                start_time = time.time()

                symetricVarianceGammaGaussLaguerre = calibrationModels(companyEquityTicker = ticker + ' Equity', horizon = int(horizon))
                nIter, assetValue, nu, sigma, distToDefault, probaToDefault = symetricVarianceGammaGaussLaguerre.SymetricVGModelGLaguerre()

                execTime = round(time.time() - start_time, 4)

                print("Le temps d'execution pour Variance Gamma Gauss Laguerre est de %s secondes" % execTime)

                if valuesVG_GL.empty:
                    valuesVG_GL = pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'nu': [nu], 'sigma': [sigma], 'distDefault':[distToDefault], 'probaDefault': [probaToDefault], 'nIter': [nIter], 'execTime': [execTime]})
                else:
                    valuesVG_GL = pd.concat([valuesVG_GL,pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'nu': [nu], 'sigma': [sigma], 'distDefault':[distToDefault], 'probaDefault': [probaToDefault], 'nIter': [nIter], 'execTime': [execTime]})], ignore_index=True)
            except:
                print(f"Pas de données sur {ticker}")

        print(valuesVG_GL)

    elif model =='VG_Lewis':

        #####################################################
        ############# Variance GAMMA Lewis Lipton ###########
        #####################################################
        
        print("\n" + "#"*120 + "\n")
        horizon = input("Entrez l'horizon (en années) auquel on s'intérèsse : ")
        print("\n" + "#"*120 + "\n")

        valuesVGLewis = pd.DataFrame({'ticker':[], 'VA': [], 'nu': [], 'sigma': [], 'distDefault':[], 'probaDefault':[],'nIter': [], 'execTime': []})
        tickers = ['SAP GY', 'MRK GY', 'AI FP', 'SU FP', 'CRH LN', 'DAI GY', 'VIE FP', 'SRG IM', 'AMP IM', 'FR FP', 'EO FP', 'GET FP', 'LHA GY', 'PIA IM', 'CO FP']

        for i in range(len(tickers)):

            ticker = tickers[i]

            print("\n" + "#"*120 + "\n")
            print(f"La boîte à laquelle on s'intérèsse est : {ticker}")
            print("\n" + "#"*120 + "\n")

            try:    
                start_time = time.time()

                symetricVarianceGammaLewis = calibrationModels(companyEquityTicker = ticker + ' Equity', horizon = int(horizon))
                nIter, assetValue, nu, sigma, distToDefault, probaToDefault = symetricVarianceGammaLewis.SymetricVGLewisLipton()

                execTime = round(time.time() - start_time, 4)

                print("Le temps d'execution pour Variance Gamma Lewis Litpon est de %s secondes" % execTime)

                if valuesVGLewis.empty:
                    valuesVGLewis = pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'nu': [nu], 'sigma': [sigma], 'distDefault':[distToDefault], 'probaDefault': [probaToDefault], 'nIter': [nIter], 'execTime': [execTime]})
                else:
                    valuesVGLewis = pd.concat([valuesVGLewis,pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'nu': [nu], 'sigma': [sigma], 'distDefault':[distToDefault], 'probaDefault': [probaToDefault], 'nIter': [nIter], 'execTime': [execTime]})], ignore_index=True)
            except:
                print(f"Pas de données sur {ticker}")

        print(valuesVGLewis)

    elif model =='VG_CM':

        #####################################################
        ############# Variance GAMMA Carr Madan #############
        #####################################################
        
        print("\n" + "#"*120 + "\n")
        horizon = input("Entrez l'horizon (en années) auquel on s'intérèsse : ")
        print("\n" + "#"*120 + "\n")

        valuesVGCarr = pd.DataFrame({'ticker':[], 'VA': [], 'nu': [], 'sigma': [], 'distDefault':[], 'probaDefault':[],'nIter': [], 'execTime': []})
        tickers = ['SAP GY', 'MRK GY', 'AI FP', 'SU FP', 'CRH LN', 'DAI GY', 'VIE FP', 'SRG IM', 'AMP IM', 'FR FP', 'EO FP', 'GET FP', 'LHA GY', 'PIA IM', 'CO FP']

        for i in range(len(tickers)):

            ticker = tickers[i]

            print("\n" + "#"*120 + "\n")
            print(f"La boîte à laquelle on s'intérèsse est : {ticker}")
            print("\n" + "#"*120 + "\n")

            try:    
                start_time = time.time()

                symetricVarianceGammaCM = calibrationModels(companyEquityTicker = ticker + ' Equity', horizon = int(horizon))
                nIter, assetValue, nu, sigma, distToDefault, probaToDefault = symetricVarianceGammaCM.SymetricVGCarrMadan()

                execTime = round(time.time() - start_time, 4)

                print("Le temps d'execution pour Variance Gamma  Carr Madan est de %s secondes" % execTime)

                if valuesVGCarr.empty:
                    valuesVGCarr = pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'nu': [nu], 'sigma': [sigma], 'distDefault':[distToDefault], 'probaDefault': [probaToDefault], 'nIter': [nIter], 'execTime': [execTime]})
                else:
                    valuesVGCarr = pd.concat([valuesVGCarr,pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'nu': [nu], 'sigma': [sigma], 'distDefault':[distToDefault], 'probaDefault': [probaToDefault], 'nIter': [nIter], 'execTime': [execTime]})], ignore_index=True)
            except:
                print(f"Pas de données sur {ticker}")

        print(valuesVGCarr)

    elif model =='VG_naive':

        #####################################################
        ############### Variance GAMMA Naive ################
        #####################################################
        
        print("\n" + "#"*120 + "\n")
        horizon = input("Entrez l'horizon (en années) auquel on s'intérèsse : ")
        print("\n" + "#"*120 + "\n")

        valuesVGNaive = pd.DataFrame({'ticker':[], 'VA': [], 'nu': [], 'sigma': [], 'distDefault':[], 'probaDefault':[],'nIter': [], 'execTime': []})
        tickers = ['SAP GY', 'MRK GY', 'AI FP', 'SU FP', 'CRH LN', 'DAI GY', 'VIE FP', 'SRG IM', 'AMP IM', 'FR FP', 'EO FP', 'GET FP', 'LHA GY', 'PIA IM', 'CO FP']

        for i in range(len(tickers)):

            ticker = tickers[i]

            print("\n" + "#"*120 + "\n")
            print(f"La boîte à laquelle on s'intérèsse est : {ticker}")
            print("\n" + "#"*120 + "\n")

            try:   
                start_time = time.time()

                symetricVarianceGammaNaive = calibrationModels(companyEquityTicker = ticker + ' Equity', horizon = int(horizon))
                nIter, assetValue, nu, sigma, distToDefault, probaToDefault = symetricVarianceGammaNaive.SymetricVGNaiveMethod()

                execTime = round(time.time() - start_time, 4)

                print("Le temps d'execution pour Variance Gamma Naive est de %s secondes" % execTime)

                if valuesVGNaive.empty:
                    valuesVGNaive = pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'nu': [nu], 'sigma': [sigma], 'distDefault':[distToDefault], 'probaDefault': [probaToDefault], 'nIter': [nIter], 'execTime': [execTime]})
                else:
                    valuesVGNaive = pd.concat([valuesVGNaive,pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'nu': [nu], 'sigma': [sigma], 'distDefault':[distToDefault], 'probaDefault': [probaToDefault], 'nIter': [nIter], 'execTime': [execTime]})], ignore_index=True)
            except:
                print(f"Pas de données sur {ticker}")

        print(valuesVGNaive)

    elif model =='VG_asymCM':

        #####################################################
        ########## Asym Variance GAMMA Carr Madan ###########
        #####################################################
        
        print("\n" + "#"*120 + "\n")
        horizon = input("Entrez l'horizon (en années) auquel on s'intérèsse : ")
        print("\n" + "#"*120 + "\n")

        riskNeutral = input("Proba physiques (False) ou proba risque neutre (True)? ")
        print("\n" + "#"*120 + "\n")
        if riskNeutral == 'False':
            riskNeutral = False
        else:
            riskNeutral = True

        valuesAsyVGCarr = pd.DataFrame({'ticker':[], 'VA': [], 'nu': [], 'sigma': [], 'theta': [], 'distDefault':[], 'probaDefault':[],'nIter': [], 'execTime': []})
        tickers = ['SAP GY', 'MRK GY', 'AI FP', 'SU FP', 'CRH LN', 'DAI GY', 'VIE FP', 'SRG IM', 'AMP IM', 'FR FP', 'EO FP', 'GET FP', 'LHA GY', 'PIA IM', 'CO FP']
        #tickers = ['DAI GY','SRG IM', 'FR FP', 'LHA GY']
        #tickers      = ['CRH LN']

        for i in range(len(tickers)):

            ticker = tickers[i]

            print("\n" + "#"*120 + "\n")
            print(f"La boîte à laquelle on s'intérèsse est : {ticker}")
            print("\n" + "#"*120 + "\n")

            try:    
                start_time = time.time()

                asymetricVarianceGammaCM = calibrationModels(companyEquityTicker = ticker + ' Equity', horizon = int(horizon), riskNeutral = riskNeutral)
                nIter, assetValue, nu, sigma, theta, distToDefault, probaToDefault = asymetricVarianceGammaCM.asymetricVGCarrMadan()

                execTime = round(time.time() - start_time, 4)

                print("Le temps d'execution pour Asymétrique Variance Gamma Carr Madan est de %s secondes" % execTime)

                if valuesAsyVGCarr.empty:
                    valuesAsyVGCarr = pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'nu': [nu], 'sigma': [sigma], 'theta': [theta], 'distDefault':[distToDefault], 'probaDefault': [probaToDefault], 'nIter': [nIter], 'execTime': [execTime]})
                else:
                    valuesAsyVGCarr = pd.concat([valuesAsyVGCarr,pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'nu': [nu], 'sigma': [sigma], 'theta': [theta], 'distDefault':[distToDefault], 'probaDefault': [probaToDefault], 'nIter': [nIter], 'execTime': [execTime]})], ignore_index=True)
            except Exception as e:
                print(e)
                print(f"Pas de données sur {ticker}")

        print(valuesAsyVGCarr)

    elif model =='VG_asymNaive':

        #####################################################
        ############ Asym Variance GAMMA Naive  #############
        #####################################################
        
        print("\n" + "#"*120 + "\n")
        horizon = input("Entrez l'horizon (en années) auquel on s'intérèsse : ")
        print("\n" + "#"*120 + "\n")

        riskNeutral = input("Proba physiques (False) ou proba risque neutre (True)? ")
        print("\n" + "#"*120 + "\n")
        if riskNeutral == 'False':
            riskNeutral = False
        else:
            riskNeutral = True

        valuesAsyVGNaive= pd.DataFrame({'ticker':[], 'VA': [], 'nu': [], 'sigma': [], 'theta': [], 'distDefault':[], 'probaDefault':[],'nIter': [], 'execTime': []})
        tickers = ['SAP GY', 'MRK GY', 'AI FP', 'SU FP', 'CRH LN', 'DAI GY', 'VIE FP', 'SRG IM', 'AMP IM', 'FR FP', 'EO FP', 'GET FP', 'LHA GY', 'PIA IM', 'CO FP']
        #tickers = ['CRH LN','DAI GY','SRG IM', 'FR FP', 'LHA GY']
        #tickers      = ['DAI GY']

        for i in range(len(tickers)):

            ticker = tickers[i]

            print("\n" + "#"*120 + "\n")
            print(f"La boîte à laquelle on s'intérèsse est : {ticker}")
            print("\n" + "#"*120 + "\n")

            try:    
                start_time = time.time()

                asymetricVarianceGammaNaive = calibrationModels(companyEquityTicker = ticker + ' Equity', horizon = int(horizon), riskNeutral = riskNeutral)
                nIter, assetValue, nu, sigma, theta, distToDefault, probaToDefault = asymetricVarianceGammaNaive.asymetricVGNaive()

                execTime = round(time.time() - start_time, 4)

                print("Le temps d'execution pour Asymétrique Variance Gamma Naive est de %s secondes" % execTime)

                if valuesAsyVGNaive.empty:
                    valuesAsyVGNaive = pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'nu': [nu], 'sigma': [sigma], 'theta': [theta], 'distDefault':[distToDefault], 'probaDefault': [probaToDefault], 'nIter': [nIter], 'execTime': [execTime]})
                else:
                    valuesAsyVGNaive = pd.concat([valuesAsyVGNaive,pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'nu': [nu], 'sigma': [sigma], 'theta': [theta], 'distDefault':[distToDefault], 'probaDefault': [probaToDefault], 'nIter': [nIter], 'execTime': [execTime]})], ignore_index=True)
            except Exception as e:
                print(e)
                print(f"Pas de données sur {ticker}")

        print(valuesAsyVGNaive)

    elif model =='bilateral':

        #####################################################
        ############## Bilateral GAMMA Naive  ###############
        #####################################################
        
        print("\n" + "#"*120 + "\n")
        horizon = input("Entrez l'horizon (en années) auquel on s'intérèsse : ")
        print("\n" + "#"*120 + "\n")

        riskNeutral = input("Proba physiques (False) ou proba risque neutre (True)? ")
        print("\n" + "#"*120 + "\n")
        if riskNeutral == 'False':
            riskNeutral = False
        else:
            riskNeutral = True

        valuesBilatGamma = pd.DataFrame({'ticker':[], 'VA': [], 'alpha+': [], 'alpha-': [], 'lambda+': [],'lambda-': [], 'distDefault':[], 'probaDefault':[],'nIter': [], 'execTime': []})
        tickers = ['SAP GY', 'MRK GY', 'AI FP', 'SU FP', 'CRH LN', 'DAI GY', 'VIE FP', 'SRG IM', 'AMP IM', 'FR FP', 'EO FP', 'GET FP', 'LHA GY', 'PIA IM', 'CO FP']
        #tickers = ['CRH LN']#,'DAI GY','SRG IM', 'FR FP', 'LHA GY']
        #tickers      = ['CO FP']

        for i in range(len(tickers)):

            ticker = tickers[i]

            print("\n" + "#"*120 + "\n")
            print(f"La boîte à laquelle on s'intérèsse est : {ticker}")
            print("\n" + "#"*120 + "\n")

            try:    
                start_time = time.time()

                bilateralGamma = calibrationModels(companyEquityTicker = ticker + ' Equity', horizon = int(horizon), riskNeutral = riskNeutral)
                nIter, assetValue, alphaP, alphaM, lambdaP, lambdaM, distToDefault, probaToDefault = bilateralGamma.bilateralGammaNaive()

                execTime = round(time.time() - start_time, 4)

                print("Le temps d'execution pour Bilateral Gamma est de %s secondes" % execTime)

                if valuesBilatGamma.empty:
                    valuesBilatGamma = pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'alpha+': [alphaP], 'alpha-': [alphaM], 'lambda+': [lambdaP], 'lambda-': [lambdaM], 'distDefault':[distToDefault], 'probaDefault': [probaToDefault], 'nIter': [nIter], 'execTime': [execTime]})
                else:
                    valuesBilatGamma = pd.concat([valuesBilatGamma,pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'alpha+': [alphaP], 'alpha-': [alphaM], 'lambda+': [lambdaP], 'lambda-': [lambdaM], 'distDefault':[distToDefault], 'probaDefault': [probaToDefault], 'nIter': [nIter], 'execTime': [execTime]})], ignore_index=True)
            
            except Exception as e:
                print(e)
                print(f"Pas de données sur {ticker}")

        print(valuesBilatGamma)

    elif model =='bilateralCM':

        #####################################################
        ############ Bilateral GAMMA Carr Madan  ############
        #####################################################
        
        print("\n" + "#"*120 + "\n")
        horizon = input("Entrez l'horizon (en années) auquel on s'intérèsse : ")
        print("\n" + "#"*120 + "\n")

        riskNeutral = input("Proba physiques (False) ou proba risque neutre (True)? ")
        print("\n" + "#"*120 + "\n")
        if riskNeutral == 'False':
            riskNeutral = False
        else:
            riskNeutral = True

        valuesBilatGamma = pd.DataFrame({'ticker':[], 'VA': [], 'alpha+': [], 'alpha-': [], 'lambda+': [],'lambda-': [], 'distDefault':[], 'probaDefault':[],'nIter': [], 'execTime': []})
        tickers = ['SAP GY', 'MRK GY', 'AI FP', 'SU FP', 'CRH LN', 'DAI GY', 'VIE FP', 'SRG IM', 'AMP IM', 'FR FP', 'EO FP', 'GET FP', 'LHA GY', 'PIA IM', 'CO FP']
        #tickers = ['CRH LN']#,'DAI GY','SRG IM', 'FR FP', 'LHA GY']
        #tickers      = ['FR FP']

        for i in range(len(tickers)):

            ticker = tickers[i]

            print("\n" + "#"*120 + "\n")
            print(f"La boîte à laquelle on s'intérèsse est : {ticker}")
            print("\n" + "#"*120 + "\n")

            try:    
                start_time = time.time()

                bilateralGamma = calibrationModels(companyEquityTicker = ticker + ' Equity', horizon = int(horizon), riskNeutral = riskNeutral)
                nIter, assetValue, alphaP, alphaM, lambdaP, lambdaM, distToDefault, probaToDefault = bilateralGamma.bilateralGammaCarrMadan()

                execTime = round(time.time() - start_time, 4)

                print("Le temps d'execution pour Bilateral Gamma est de %s secondes" % execTime)

                if valuesBilatGamma.empty:
                    valuesBilatGamma = pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'alpha+': [alphaP], 'alpha-': [alphaM], 'lambda+': [lambdaP], 'lambda-': [lambdaM], 'distDefault':[distToDefault], 'probaDefault': [probaToDefault], 'nIter': [nIter], 'execTime': [execTime]})
                else:
                    valuesBilatGamma = pd.concat([valuesBilatGamma,pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'alpha+': [alphaP], 'alpha-': [alphaM], 'lambda+': [lambdaP], 'lambda-': [lambdaM], 'distDefault':[distToDefault], 'probaDefault': [probaToDefault], 'nIter': [nIter], 'execTime': [execTime]})], ignore_index=True)
            
            except Exception as e:
                print(e)
                print(f"Pas de données sur {ticker}")

        print(valuesBilatGamma)

    elif model =='bilateralLL':

        #####################################################
        ########### Bilateral GAMMA Lewis Lipton  ###########
        #####################################################
        
        print("\n" + "#"*120 + "\n")
        horizon = input("Entrez l'horizon (en années) auquel on s'intérèsse : ")
        print("\n" + "#"*120 + "\n")

        riskNeutral = input("Proba physiques (False) ou proba risque neutre (True)? ")
        print("\n" + "#"*120 + "\n")
        if riskNeutral == 'False':
            riskNeutral = False
        else:
            riskNeutral = True

        valuesBilatGamma = pd.DataFrame({'ticker':[], 'VA': [], 'alpha+': [], 'alpha-': [], 'lambda+': [],'lambda-': [], 'distDefault':[], 'probaDefault':[],'nIter': [], 'execTime': []})
        tickers = ['SAP GY', 'MRK GY', 'AI FP', 'SU FP', 'CRH LN', 'DAI GY', 'VIE FP', 'SRG IM', 'AMP IM', 'FR FP', 'EO FP', 'GET FP', 'LHA GY', 'PIA IM', 'CO FP']
        #tickers = ['CRH LN']#,'DAI GY','SRG IM', 'FR FP', 'LHA GY']
        #tickers      = ['DAI GY','CRH LN']

        for i in range(len(tickers)):

            ticker = tickers[i]

            print("\n" + "#"*120 + "\n")
            print(f"La boîte à laquelle on s'intérèsse est : {ticker}")
            print("\n" + "#"*120 + "\n")

            try:    
                start_time = time.time()

                bilateralGamma = calibrationModels(companyEquityTicker = ticker + ' Equity', horizon = int(horizon), riskNeutral = riskNeutral)
                nIter, assetValue, alphaP, alphaM, lambdaP, lambdaM, distToDefault, probaToDefault = bilateralGamma.bilateralGammaLewisLipton()

                execTime = round(time.time() - start_time, 4)

                print("Le temps d'execution pour Bilateral Gamma est de %s secondes" % execTime)

                if valuesBilatGamma.empty:
                    valuesBilatGamma = pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'alpha+': [alphaP], 'alpha-': [alphaM], 'lambda+': [lambdaP], 'lambda-': [lambdaM], 'distDefault':[distToDefault], 'probaDefault': [probaToDefault], 'nIter': [nIter], 'execTime': [execTime]})
                else:
                    valuesBilatGamma = pd.concat([valuesBilatGamma,pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'alpha+': [alphaP], 'alpha-': [alphaM], 'lambda+': [lambdaP], 'lambda-': [lambdaM], 'distDefault':[distToDefault], 'probaDefault': [probaToDefault], 'nIter': [nIter], 'execTime': [execTime]})], ignore_index=True)
            
            except Exception as e:
                print(e)
                print(f"Pas de données sur {ticker}")

        print(valuesBilatGamma)

    else:
        print("Vous n'avez pas choisi un modèle parmis les suggestions, veuillez réessayer")

