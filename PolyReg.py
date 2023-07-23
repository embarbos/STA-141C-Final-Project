# Polynomial regression function
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import time
from scipy import linalg
from sklearn.metrics import mean_squared_error

def PolyReg(x_train,y_train,x_test,y_test,degree,method):
    # Polynomial regression
    # Set the maximum degree of the polynomial to fit
    n_degree=degree
    
    start = time.perf_counter()
    # Create polynomial features for the training and testing data
    poly = PolynomialFeatures(degree)
    x_poly_train = poly.fit_transform(x_train)
    x_poly_test = poly.fit_transform(x_test)
    if method == 'QR':
        # Prepare decompositions
        XX = x_poly_train.T @ x_poly_train
        XY = x_poly_train.T @ y_train
        Q, R = np.linalg.qr(XX)
        RB = Q.T@XY
        Bhat_QR = linalg.solve_triangular(R,RB,lower=False)
        
        # Predictions
        pred_train =  x_poly_train @ Bhat_QR
        pred_test = x_poly_test @ Bhat_QR
    
        # Fit a linear regression model on the polynomial features
        #LR = LinearRegression()
        #LR.fit(x_poly_train, y_train)
    
        # Predict on the training data and calculate the RMSE
        #pred1 = LR.predict(x_poly_train)
        Trr = mean_squared_error(y_train, pred_train)
    
        # Predict on the testing data and calculate the RMSE
        #pred2 = LR.predict(x_poly_test)
        Tss = mean_squared_error(y_test, pred_test)
    if method == 'LU':
        # Create polynomial features for the training and testing data
        # Prepare decompositions
        XX = x_poly_train.T @ x_poly_train
        XY = x_poly_train.T @ y_train
        P, L, U = linalg.lu(XX)
        LZ= P.T @ XY
        Z = linalg.solve_triangular(L,LZ,lower=True)
        Bhat_LU = linalg.solve_triangular(U,Z,lower=False)
        
        # Predictions
        pred_train =  x_poly_train @ Bhat_LU
        pred_test = x_poly_test @ Bhat_LU
    
        # Fit a linear regression model on the polynomial features
        #LR = LinearRegression()
        #LR.fit(x_poly_train, y_train)
    
        # Predict on the training data and calculate the RMSE
        #pred1 = LR.predict(x_poly_train)
        Trr = mean_squared_error(y_train, pred_train)
    
        # Predict on the testing data and calculate the RMSE
        #pred2 = LR.predict(x_poly_test)
        Tss = mean_squared_error(y_test, pred_test)
#    if method=='CHOLESKY':
#        for i in range(2, n_degree):
            # Create polynomial features for the training and testing data
#            poly = PolynomialFeatures(degree=i)
#            x_poly_train = poly.fit_transform(x_train)
#            x_poly_test = poly.fit_transform(x_test)
            
            # Prepare decompositions
#            XX = x_poly_train.T @ x_poly_train
#            XY = x_poly_train.T @ y_train
#            print (np.linalg.eigvalsh(XX))
#            print(np.linalg.eigvalsh(XY))
#            c,low = scipy.linalg.cho_factor(XX)
#            Bhat_cho = scipy.linalg.cho_solve((c,low),XY)
#            print(Bhat_cho)
            
            # Predictions
#            pred_train =  x_poly_train @ Bhat_cho
#            pred_test = x_poly_test @ Bhat_cho
#            print(pred_train.shape,pred_test.shape)
        
            # Fit a linear regression model on the polynomial features
            #LR = LinearRegression()
            #LR.fit(x_poly_train, y_train)
        
            # Predict on the training data and calculate the MSE
            #pred1 = LR.predict(x_poly_train)
#            Trr.append(mean_squared_error(y_train, pred_train))
        
            # Predict on the testing data and calculate the RMSE
            #pred2 = LR.predict(x_poly_test)
#            Tss.append(mean_squared_error(y_test, pred_test))  
    end = time.perf_counter()
    # Plot the MSE of the training and testing errors for different degrees of the polynomial
    total = end - start
    return (Trr, Tss, total)