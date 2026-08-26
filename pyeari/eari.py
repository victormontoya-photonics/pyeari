import numpy as np
import cv2
from ri import guidedfilter, bilinear_RB, ri,standardize_input
def MEARI(MPFA):
    eps=1e-32
    Fn=np.array([[1,2,1],[1,2,1],[0,0,0]], dtype=np.float32)/8
    Fs=np.array([[0,0,0],[1,2,1],[1,2,1]], dtype=np.float32)/8
    Fw=Fn.T
    Fe=Fs.T

    Hn=np.array([[-1,2,-1],[1,-2,1],[0,0,0]], dtype=np.float32)/2
    Hs=np.array([[0,0,0],[1,-2,1],[-1,2,-1]], dtype=np.float32)/2
    Hw=Hn.T
    He=Hs.T

    Mn=np.ones((5,5), dtype=np.float32)/15
    Mn[3:,:]=0
    Ms=Mn[::-1,:]
    Mw=Mn.T
    Me=Ms.T

    MPFA32 = standardize_input(MPFA)

    Xn = cv2.filter2D(MPFA32, -1, Fn, borderType=cv2.BORDER_REFLECT_101)
    Xs = cv2.filter2D(MPFA32, -1, Fs, borderType=cv2.BORDER_REFLECT_101)
    Xe = cv2.filter2D(MPFA32, -1, Fe, borderType=cv2.BORDER_REFLECT_101)
    Xw = cv2.filter2D(MPFA32, -1, Fw, borderType=cv2.BORDER_REFLECT_101)

    Dn = cv2.filter2D(MPFA32, -1, Hn, borderType=cv2.BORDER_REFLECT_101)
    Ds = cv2.filter2D(MPFA32, -1, Hs, borderType=cv2.BORDER_REFLECT_101)
    De = cv2.filter2D(MPFA32, -1, He, borderType=cv2.BORDER_REFLECT_101)
    Dw = cv2.filter2D(MPFA32, -1, Hw, borderType=cv2.BORDER_REFLECT_101)

    sDn = cv2.filter2D(np.abs(Dn), -1, Mn, borderType=cv2.BORDER_REFLECT_101)
    sDs = cv2.filter2D(np.abs(Ds), -1, Ms, borderType=cv2.BORDER_REFLECT_101)
    sDe = cv2.filter2D(np.abs(De), -1, Me, borderType=cv2.BORDER_REFLECT_101)
    sDw = cv2.filter2D(np.abs(Dw), -1, Mw, borderType=cv2.BORDER_REFLECT_101)

    Wn = 1 / (sDn + eps)
    Ws = 1 / (sDs + eps)
    We = 1 / (sDe + eps)
    Ww = 1 / (sDw + eps)

    W = Wn + Ws + We + Ww

    G = (Wn*Xn + Ws*Xs + We*Xe + Ww*Xw) / W

    M = np.zeros_like(MPFA)
    M90 = np.copy(M)
    M00 = np.copy(M)
    M45 = np.copy(M)
    M135 = np.copy(M)

    M90[0::2,0::2] = 1
    M00[1::2,1::2] = 1
    M45[0::2,1::2] = 1
    M135[1::2,0::2] = 1
    
    I90_sparse = np.copy(M)
    I00_sparse = np.copy(M)
    I45_sparse = np.copy(M)
    I135_sparse = np.copy(M)
    
    I90_sparse[0::2,0::2] = MPFA[0::2,0::2]
    I00_sparse[1::2,1::2] = MPFA[1::2,1::2]
    I45_sparse[0::2,1::2] = MPFA[0::2,1::2]
    I135_sparse[1::2,0::2] = MPFA[1::2,0::2]
    
    I90_t = guidedfilter(G, I90_sparse, M90, 5, 5)
    I00_t = guidedfilter(G, I00_sparse, M00, 5, 5)
    I45_t = guidedfilter(G, I45_sparse, M45, 5, 5)
    I135_t = guidedfilter(G, I135_sparse, M135, 5, 5)

    I90_res = M90 * (MPFA - I90_t)
    I00_res = M00 * (MPFA - I00_t)
    I45_res = M45 * (MPFA - I45_t)
    I135_res = M135 * (MPFA - I135_t)

    I90 = bilinear_RB(I90_res) + I90_t
    I00 = bilinear_RB(I00_res) + I00_t
    I45 = bilinear_RB(I45_res) + I45_t
    I135 = bilinear_RB(I135_res) + I135_t

    return I90, I00, I45, I135

def make_mpfa(rgb90,rgb00,rgb45,rgb135):
    R90,G90,B90=rgb90[:,:,0],rgb90[:,:,1],rgb90[:,:,2]
    R00,G00,B00=rgb00[:,:,0],rgb00[:,:,1],rgb00[:,:,2]
    R45,G45,B45=rgb45[:,:,0],rgb45[:,:,1],rgb45[:,:,2]
    R135,G135,B135=rgb135[:,:,0],rgb135[:,:,1],rgb135[:,:,2]
    x,y=rgb90.shape[:2]
    
    RMPFA=np.empty((2*x,2*y), dtype=np.float32)
    GMPFA=np.empty((2*x,2*y), dtype=np.float32)
    BMPFA=np.empty((2*x,2*y), dtype=np.float32)
    
    RMPFA[::2,::2]=R90
    RMPFA[1::2,1::2]=R00
    RMPFA[0::2,1::2]=R45
    RMPFA[1::2,0::2]=R135
    
    GMPFA[::2,::2]=G90
    GMPFA[1::2,1::2]=G00
    GMPFA[0::2,1::2]=G45
    GMPFA[1::2,0::2]=G135
    
    BMPFA[::2,::2]=B90
    BMPFA[1::2,1::2]=B00
    BMPFA[0::2,1::2]=B45
    BMPFA[1::2,0::2]=B135
    
    return RMPFA,GMPFA,BMPFA
    
def CEARI(CPFA,pattern):
    CPFA_standard = standardize_input(CPFA)
    cfa90=CPFA_standard[0::2,0::2]
    cfa00=CPFA_standard[1::2,1::2]
    cfa45=CPFA_standard[0::2,1::2]
    cfa135=CPFA_standard[1::2,0::2]
    #if speed is importnat you can use a faster color demosaicking algo
    rgb90=ri(cfa90,pattern)
    rgb00=ri(cfa00,pattern)
    rgb45=ri(cfa45,pattern)
    rgb135=ri(cfa135,pattern)
    
    RMPFA,GMPFA,BMPFA=make_mpfa(rgb90,rgb00,rgb45,rgb135)
    
    R90,R00,R45,R135=MEARI(RMPFA)
    G90,G00,G45,G135=MEARI(GMPFA)
    B90,B00,B45,B135=MEARI(BMPFA)
    
    RGB90=np.stack((R90,G90,B90),axis=-1)
    RGB00=np.stack((R00,G00,B00),axis=-1)
    RGB45=np.stack((R45,G45,B45),axis=-1)
    RGB135=np.stack((R135,G135,B135),axis=-1)
        
    RGB90=np.clip(RGB90,0,1)
    RGB00=np.clip(RGB00,0,1)
    RGB45=np.clip(RGB45,0,1)
    RGB135=np.clip(RGB135,0,1)
    return RGB90,RGB00,RGB45,RGB135
