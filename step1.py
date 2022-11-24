import pandas as pd
from scipy.stats import norm, kurtosis
from statistics import variance
from scipy import optimize, special
import scipy.integrate as integrate
import numpy as np
import math
import os


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, './Data10.xlsx')
dataEquity = pd.read_excel(filename, sheet_name='Mod Market Cap') 
dataEquity = dataEquity.set_index('Dates').loc['2019-10-28':'2020-10-13']
dataDebt = pd.read_excel(filename, sheet_name='Gross Debt').dropna()

class calibrationModels():

    def __init__(self, companyEquityTicker = 'CRH LN Equity', timePeriod = 252, horizon = 1,tolerance = 10e-5):
        
        self.companyEquityTicker_       = companyEquityTicker
        self.tolerance_                 = tolerance
        self.timePeriod_                = timePeriod #Correspond au nombre de jours de la période étudiée
        self.horizon_                   = horizon
        self.relativeTime_              = 0 #Correspond au temps relatif où on effectue le calcul de BlackScholesMerton
        self.riskFIR_                   = 0 #Risk-free interest rate

        self.sigma_A_                   = 0
        self.sigma_E_                   = 0
        self.sigma_A_history_           = [-100]

        self.rho_                       = 0
        self.lambda_                    = 0
        self.mu_                        = 0

        self.rho_history_               = [-100]
        self.lambda_history_            = [-100]
        self.mu_history_                = [-100]
        self.nu_history_= [-100]
        self.sigma_history_= [-100]

        self.asset_values_              = [] #Correspond à V_A
        self.equity_value_              = 0  #Correspond à V_E

        self.nombreIterations_          = 0
        self.companyDebt_               = dataDebt[[self.companyEquityTicker_]].iloc[0,0]
        self.companyEquityListValues_   = dataEquity[[self.companyEquityTicker_]].iloc[:,0]

    ##################################################################################################################
    ######################################### Fonctions de répartition ###############################################
    ##################################################################################################################
    def densityGaussianDistribution(self,x):
        return np.exp(-1*0.5*(x**2))/np.sqrt(2*np.pi)

    def cumulativeGaussianDistribution(self,x):
        return norm.cdf(x)

    def gamma(self,x):
        return special.gamma(x)

    def lowerIncompleteGamma(self, a, x):
        return self.gamma(a)*special.gammainc(a, x)

    def upperIncompleteGamma(self, a, x):
        return self.gamma(a)*special.gammaincc(a, x)
    
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
            self.mu_           = self.mu_history_[-1]
            
            for day in range(self.companyEquityListValues_.shape[0]):

                self.relativeTime_ = self.horizon_ + (self.timePeriod_ - day - 1)/252
                self.equity_value_ = self.companyEquityListValues_[day]
                self.asset_values_.append(optimize.newton(modelNegIG,self.companyDebt_))

            assetRelativeValue = np.diff(np.log(self.asset_values_), n = 1)
            kurtosis_           = kurtosis(assetRelativeValue, fisher = False)
            variance_           = variance(assetRelativeValue)*252

            self.lambda_history_.append(parameterLambda(variance_, kurtosis_))
            self.mu_history_.append(parameterMu(variance_, kurtosis_))
            print(f"A l'itération {self.nombreIterations_} mu = {round(self.mu_history_[-1],3)} et lambda = {round(self.lambda_history_[-1],3)} et VA = {self.asset_values_[-1]}")
            self.nombreIterations_ += 1

        self.mu_           = self.mu_history_[-1]
        self.lambda_        = self.lambda_history_[-1]

        negInvGaussianDistanceToDefault   = kI(self.asset_values_[-1])
        negInvGaussianDefaultProbability  = 1 if negInvGaussianDistanceToDefault <=0 else defaultProbability(negInvGaussianDistanceToDefault)*100

        return self.nombreIterations_, self.asset_values_[-1], self.mu_, self.lambda_, negInvGaussianDistanceToDefault, negInvGaussianDefaultProbability

    def VarianceGammaModel(self):

        def roots_Laguerre(n):
            x,_ = special.roots_laguerre(n, mu=False)
            weight=compute_weight(x,n)
            return x,weight

        def compute_weight(x,n):
            weight=[ u/(((n+1)*special.genlaguerre(n+1, 0)(u))**2) for u in x]
            return weight

        def pr(x,weight,i,t):
            numerator=weight[i]*(x[i]**(t-1))
            denominator=np.sum(np.array([weight[i]*(x[i]**(t-1)) for i in range(len(x))]))
            return numerator/denominator

        def w_(x,weight,n,sigma,nu,t):
            sum_p=0
            for i in range(n):
                sum_p+=pr(x,weight,i,t/nu)*np.exp(((sigma**2)/2)*nu*x[i])
            return -1*np.log(sum_p)/t

        def parameterNu(Kurtosis):
            return Kurtosis/3

        def d1(v,x,i,w,sigma,nu):
            num=np.log(v/self.companyDebt_)+(self.riskFIR_+w)*self.timePeriod_+(sigma**2)*nu*x[i]
            den=sigma*np.sqrt(nu*x[i])
            return num/den
        
        def d2(v,x,i,w,sigma,nu):
            return d1(v,x,i,w,sigma,nu)-sigma*np.sqrt(nu*x[i])
        
        def modelVG(v):
            n=4
            x,weight=roots_Laguerre(n)
            leftPart=v
            rightPart=self.companyDebt_*np.exp(-1*self.relativeTime_*self.riskFIR_)
            temp_left=0
            temp_right=0
            w=w_(x,weight,n,self.sigma_,self.nu_,self.relativeTime_)

            for i in range(n):
                p=pr(x,weight,i,self.relativeTime_/self.nu_)
                temp_right+=p*self.cumulativeGaussianDistribution(d2(v,x,i,w,self.sigma_,self.nu_))
                temp_left+=np.exp(w*self.relativeTime_+(0.5*(self.sigma_**2))*self.nu_*x[i])*self.cumulativeGaussianDistribution(d1(v,x,i,w,self.sigma_,self.nu_))*p
            leftPart*=temp_left
            rightPart*=temp_right

            return leftPart - rightPart 

        def f_prime(v,n):
            fprim=0
            x,weight=roots_Laguerre(n)
            w=w_(x,weight,n,self.sigma_,self.nu_,self.relativeTime_)
            for i in range(n):
                d1_=d1(v,x,i,w,self.sigma_,self.nu_)
                d2_=d2(v,x,i,w,self.sigma_,self.nu_)
                p=pr(x,weight,i,self.relativeTime_/self.nu_)
                temp=np.exp(w*self.relativeTime_+0.5*(self.sigma_**2)*self.nu_*x[i])
                temp_denum=(self.sigma_*np.sqrt(self.nu_*x[i]))
                temp*=self.cumulativeGaussianDistribution(d1_)+self.densityGaussianDistribution(d1_)/temp_denum
                temp-=self.companyDebt_*np.exp(-1*self.riskFIR_*self.relativeTime_)*self.densityGaussianDistribution(d2_)/(v*temp_denum)
                fprim+=temp*p
            return fprim

        def f_sigma_nu(x):
            temp=2/(np.sqrt(2*np.pi)*special.gamma(self.relativeTime_/self.nu_)*(((self.sigma_**2)*self.nu_)**(self.relativeTime_/self.nu_)))
            a=np.sqrt(2/self.nu_)/self.sigma_
            temp*=(np.abs(x)/a)**((self.relativeTime_/self.nu_)-0.5)
            temp*=K_alpha(((self.relativeTime_/self.nu_)-0.5),np.abs(x)*a)
            return temp

        def K_alpha(alpha,x):
            return (2**(x-2))*special.gamma((x-alpha)/2)*special.gamma((x+alpha)/2)

        def modelVG_naive(v):
            w_=np.log(1-((self.sigma_**2)*self.nu_*0.5))/self.nu_
            k_sigma_nu=np.log(v/self.companyDebt_)+self.riskFIR_*self.relativeTime_
            k_sigma_nu+=self.relativeTime_*w_
            temp=lambda x: (v*np.exp((self.riskFIR_+w_)*self.horizon_+x)-self.companyDebt_)
            return np.exp(-1*self.riskFIR_*self.relativeTime_)*integrate.quad(lambda x: f_sigma_nu(x)*temp(x), -1*k_sigma_nu, np.inf)[0]

        def f_prime_naive(v):
            w_=np.log(1-((self.sigma_**2)*self.nu_*0.5))/self.nu_
            k_sigma_nu=np.log(v/self.companyDebt_)+self.riskFIR_*self.relativeTime_
            k_sigma_nu+=self.relativeTime_*w_
            return integrate.quad(lambda x:f_sigma_nu(x)*np.exp((self.riskFIR_+w_)*self.horizon_+x),-1*k_sigma_nu, 999)[0]
        #initialisation des paramètres
        n=4
        equityRelativeValue     = np.diff(np.log(self.companyEquityListValues_), n = 1)

        self.sigma_            = np.sqrt(variance(equityRelativeValue))
        self.nu_                = parameterNu(kurtosis(equityRelativeValue))
        self.sigma_history_.append(self.sigma_)
        self.nu_history_.append(self.nu_)

        while (np.abs(self.sigma_history_[-1] - self.sigma_history_[-2]) > self.tolerance_) or (np.abs(self.nu_history_[-1] - self.nu_history_[-2]) > self.tolerance_):

            self.asset_values_  = []
            self.sigma_        = self.sigma_history_[-1]
            self.nu_           = self.nu_history_[-1]
            
            for day in range(self.companyEquityListValues_.shape[0]):

                self.relativeTime_ = self.horizon_ + (self.timePeriod_ - day - 1)/252
                self.equity_value_ = self.companyEquityListValues_[day]
                self.asset_values_.append(optimize.newton(modelVG,self.companyDebt_,fprime=lambda x:f_prime(x,n),maxiter=2000,tol=1e-6))

            assetRelativeValue = np.diff(np.log(self.asset_values_), n = 1)
            kurtosis_           = kurtosis(assetRelativeValue, fisher = False)
            variance_           = variance(assetRelativeValue)*252

            self.sigma_history_.append(np.sqrt(variance_))
            self.nu_history_.append(parameterNu( kurtosis_))
            print(f"A l'itération {self.nombreIterations_} nu = {round(self.nu_history_[-1],3)} et sigma = {round(self.sigma_history_[-1],3)} et VA = {self.asset_values_[-1]}")
            self.nombreIterations_ += 1

        self.nu_           = self.nu_history_[-1]
        self.sigma_        = self.sigma_history_[-1]

    ###    negInvGaussianDistanceToDefault   = kI(self.asset_values_[-1])
      ###  negInvGaussianDefaultProbability  = 1 if negInvGaussianDistanceToDefault <=0 else defaultProbability(negInvGaussianDistanceToDefault)*100

        return self.nombreIterations_, self.asset_values_[-1], self.nu_, self.sigma_##, negInvGaussianDistanceToDefault, negInvGaussianDefaultProbability

if __name__ == "__main__":

    print("\n" + "#"*120) 
    print("#"*120 + "\n") 
    model = input("Quel modèle voulez-vous utiliser parmis Merton, NegGamma, NegIG, VG ?  ")
    print("\n" + "#"*120 ) 
    print("#"*120 + "\n") 

    if model == 'NegGamma':

        #####################################################
        ##################### NEG GAMMA #####################
        #####################################################

        print("\n" + "#"*120 + "\n")
        horizon = input("Entrez l'horizon (en années) auquel on s'intérèsse : ")
        print("\n" + "#"*120 + "\n")   

        valuesNegGamma = pd.DataFrame({'ticker':[], 'VA': [], 'lambda': [], 'rho': [], 'distDefault':[], 'probaDefault':[], 'nIter': []})
        tickers = ['SAP GY', 'MRK GY', 'AI FP', 'SU FP', 'CRH LN', 'DAI GY', 'VIE FP', 'SRG IM', 'AMP IM', 'FR FP', 'EO FP', 'GET FP', 'LHA GY', 'PIA IM', 'CO FP']

        for i in range(len(tickers)):

            ticker = tickers[i]

            print("\n" + "#"*120 + "\n")
            print(f"La boîte à laquelle on s'intérèsse est : {ticker}")
            print("\n" + "#"*120 + "\n")

            try:    
                NegGamma = calibrationModels(companyEquityTicker = ticker + ' Equity', horizon = int(horizon))
                nIter, assetValue, rho, lambd, distToDefault, defaultProba = NegGamma.NegativeGammaModel()
                valuesNegGamma = valuesNegGamma.append({'ticker':ticker, 'VA':assetValue, 'lambda': lambd, 'rho': rho, 'distDefault':distToDefault, 'probaDefault':defaultProba, 'nIter': nIter}, ignore_index = True)
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

        valuesNegIG = pd.DataFrame({'ticker':[], 'VA': [], 'mu': [], 'lambda': [], 'distDefault':[], 'probaDefault':[], 'nIter': []})
        tickers = ['SAP GY', 'MRK GY', 'AI FP', 'SU FP', 'CRH LN', 'DAI GY', 'VIE FP', 'SRG IM', 'AMP IM', 'FR FP', 'EO FP', 'GET FP', 'LHA GY', 'PIA IM', 'CO FP']

        for i in range(len(tickers)):

            ticker = tickers[i]

            print("\n" + "#"*120 + "\n")
            print(f"La boîte à laquelle on s'intérèsse est : {ticker}")
            print("\n" + "#"*120 + "\n")

            try:    
                NegIG = calibrationModels(companyEquityTicker = ticker + ' Equity', horizon = int(horizon))
                nIter, assetValue, mu, lambd, distToDefault, defaultProba = NegIG.NegInvGaussianModel()
                valuesNegIG = valuesNegIG.append({'ticker':ticker, 'VA':assetValue, 'lambda': lambd, 'mu': mu, 'distDefault':distToDefault, 'probaDefault':defaultProba, 'nIter': nIter}, ignore_index = True)
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

        valuesMerton = pd.DataFrame({'ticker':[], 'VA': [], 'sigmaA': [], 'distDefault':[], 'probaDefault':[], 'nIter': []})
        tickers = ['SAP GY', 'MRK GY', 'AI FP', 'SU FP', 'CRH LN', 'DAI GY', 'VIE FP', 'SRG IM', 'AMP IM', 'FR FP', 'EO FP', 'GET FP', 'LHA GY', 'PIA IM', 'CO FP']

        for i in range(len(tickers)):

            ticker = tickers[i]

            print("\n" + "#"*120 + "\n")
            print(f"La boîte à laquelle on s'intérèsse est : {ticker}")
            print("\n" + "#"*120 + "\n")

            try:    
                merton = calibrationModels(companyEquityTicker = ticker + ' Equity', horizon = int(horizon))
                nIter, assetValue, sigmaA, distToDefault, defaultProba = merton.BlackScholesMertonModel()
                valuesMerton = valuesMerton.append({'ticker':ticker, 'VA':assetValue, 'sigmaA': sigmaA, 'distDefault':distToDefault, 'probaDefault':defaultProba, 'nIter': nIter}, ignore_index = True)
            except:
                print(f"Pas de données sur {ticker}")
        print(valuesMerton)

    elif model =='VG':

            #####################################################
            #################### Variance GAMMA ####################
            #####################################################
            
            print("\n" + "#"*120 + "\n")
            horizon = input("Entrez l'horizon (en années) auquel on s'intérèsse : ")
            print("\n" + "#"*120 + "\n")   

            valuesVG = pd.DataFrame({'ticker':[], 'VA': [], 'mu': [], 'lambda': [], 'distDefault':[], 'probaDefault':[], 'nIter': []})
            tickers = ['CRH LN']

            for i in range(len(tickers)):

                ticker = tickers[i]

                print("\n" + "#"*120 + "\n")
                print(f"La boîte à laquelle on s'intérèsse est : {ticker}")
                print("\n" + "#"*120 + "\n")

                try:    
                    VG = calibrationModels(companyEquityTicker = ticker + ' Equity', horizon = int(horizon))
                   ## nIter, assetValue, nu, sigma, distToDefault, defaultProba = VG.modelVG()
                except:
                    print(f"Pas de données sur {ticker}")
                nIter, assetValue, nu, sigma = VG.VarianceGammaModel()

                valuesVG = valuesVG.append({'ticker':ticker, 'VA':assetValue, 'sigma': sigma, 'nu': nu, 'nIter': nIter}, ignore_index = True)

            print(valuesVG)

    else:
        print("Vous n'avez pas choisi un modèle parmis les suggestions, veuillez réessayer")

