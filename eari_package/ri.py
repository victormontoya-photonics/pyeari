import numpy as np
from scipy import ndimage as nd
import cv2

offsets = {
        'RGGB': {'R': (0,0), 'Gr': (0,1), 'Gb': (1,0), 'B': (1,1)},
        'BGGR': {'R': (1,1), 'Gr': (1,0), 'Gb': (0,1), 'B': (0,0)},
        'GRBG': {'R': (0,1), 'Gr': (0,0), 'Gb': (1,1), 'B': (1,0)},
        'GBRG': {'R': (1,0), 'Gr': (1,1), 'Gb': (0,0), 'B': (0,1)},
    }
def standardize_input(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype == np.float32:
        return img
    else:
        return img.astype(np.float32)
        
def getRGB_known(cfa,pattern):
    R=np.zeros(cfa.shape, dtype=np.float32)
    G=np.copy(R)
    B=np.copy(R)
    channels={'R':R,'G':G,'B':B}
    for ch, (r0, c0) in offsets[pattern].items():
        channels[ch[0]][r0::2,c0::2]=cfa[r0::2,c0::2]
    return R,G,B

def make_cfa(img_path, pattern):
    channel={'B':0,'Gr':1,'Gb':1,'R':2}
    bgr_img=cv2.imread(img_path).astype(np.float32)/255.0
    cfa=np.zeros(bgr_img.shape[:2], dtype=np.float32) 
    for ch, (r0, c0) in offsets[pattern].items():
        cfa[r0::2,c0::2]=bgr_img[r0::2,c0::2,channel[ch]]
    return cfa

def bilinear_1D(C_known_h,C_known_v,channel,pattern):
    C_h=np.zeros_like(C_known_h)
    C_v=np.zeros_like(C_known_v)
    r,c=offsets[pattern][channel]
    
    kernel_h = np.array([[0.5,1.0,0.5]], dtype=np.float32)
    kernel_v=kernel_h.T

    C_h[r::2,:]=cv2.filter2D(C_known_h[r::2,:].astype(np.float32), -1, kernel_h, borderType=cv2.BORDER_REFLECT_101)
    C_v[:,c::2]=cv2.filter2D(C_known_v[:,c::2].astype(np.float32), -1, kernel_v, borderType=cv2.BORDER_REFLECT_101)

    return C_h,C_v

def get_residuals(C_known,C_tent,channel,pattern):
    r,c=offsets[pattern][channel]
    C_res=np.zeros(C_known.shape, dtype=np.float32)
    C_res[r::2,c::2]=C_known[r::2,c::2]-C_tent[r::2,c::2]
    return C_res

def overlap(C,channel,pattern):
    r,c=offsets[pattern][channel]
    C_overlap=np.zeros(C.shape, dtype=np.float32)
    C_overlap[r::2,c::2]=C[r::2,c::2]
    return C_overlap
    
def half_gaussian_kernel(size, sigma):
    x = np.arange(size) 
    kernel = np.exp(-x**2 / (2 * sigma**2))
    return kernel / kernel.sum()
    
def get_delta(delta_h,delta_v,channel,pattern,N=5,sigma=1e+8):
    delta_h_pad=np.pad(delta_h,((0,0),(1,1)),mode='symmetric')
    delta_v_pad=np.pad(delta_v,((1,1),(0,0)),mode='symmetric')
    
    Dh=np.abs(delta_h_pad[:,:-2]-delta_h_pad[:,2:])
    Dv=np.abs(delta_v_pad[:-2,:]-delta_v_pad[2:,:])
    eps=1e-5

    wn = 1/(nd.uniform_filter(Dv, size=N, mode='mirror', origin=(2, 0))+eps)**2
    ws = 1/(nd.uniform_filter(Dv, size=N, mode='mirror', origin=(-2, 0))+eps)**2
    ww = 1/(nd.uniform_filter(Dh, size=N, mode='mirror', origin=(0, 2))+eps)**2
    we = 1/(nd.uniform_filter(Dh, size=N, mode='mirror', origin=(0, -2))+eps)**2
    
    wt=wn+ws+we+ww
    f=half_gaussian_kernel(N, sigma)
    
    delta_s=ws*nd.correlate1d(delta_v,f,origin=-2,axis=0,mode='mirror')
    delta_e=we*nd.correlate1d(delta_h,f,origin=-2,axis=1,mode='mirror')
    delta_w=ww*nd.correlate1d(delta_h,f[::-1],origin=2,axis=1,mode='mirror')
    delta_n=wn*nd.correlate1d(delta_v,f[::-1],origin=2,axis=0,mode='mirror')
    
    delta=np.zeros(delta_h.shape, dtype=np.float32)
    r,c=offsets[pattern][channel]
    delta[r::2,c::2]=((delta_n+delta_s+delta_e+delta_w)/wt)[r::2,c::2]

    return delta
    
def guidedfilter(I,p,M,h,v,eps=0):
    size_v=2*v+1 if v!=0 else 1
    size_h=2*h+1 if h!=0 else 1
    
    I32 = I.astype(np.float32, copy=False)
    p32 = p.astype(np.float32, copy=False)
    M32 = M.astype(np.float32, copy=False)

    def boxfilter(img):
        return cv2.boxFilter(img, cv2.CV_32F, (size_h, size_v), normalize=False, borderType=cv2.BORDER_REFLECT_101)

    N = boxfilter(M32)
    N[N==0]=1
    N2 = boxfilter(np.ones_like(I32))
    
    mean_I = boxfilter(I32*M32)/N
    mean_p = boxfilter(p32)/N      
    mean_Ip = boxfilter(I32*p32)/N  
    
    cov_Ip = mean_Ip - mean_I*mean_p
    
    mean_II = boxfilter(I32*I32*M32)/N
    var_I = mean_II - mean_I*mean_I
    
    th=1e-5 
    var_I = np.maximum(var_I, th)
    
    a = cov_Ip/(var_I+eps)
    b = mean_p - a*mean_I
    
    mean_a = boxfilter(a)/N2
    mean_b = boxfilter(b)/N2
    
    q = mean_a*I32 + mean_b
    return q

def guided_1D(G_d,C_known,d,channel,pattern,ep=0):
    r,c=offsets[pattern][channel]
    M=np.zeros_like(C_known)
    M[r::2,c::2]=1.0
    p=C_known*M
    
    h=5 if d==1 else 0
    v=5 if d==0 else 0
    
    C_tent=guidedfilter(G_d,p,M,h,v,ep)
    
    C_d=np.zeros_like(G_d)
    if d==0:
        C_d[:,c::2]=C_tent[:,c::2]
    elif d==1:
        C_d[r::2,:]=C_tent[r::2,:]
        
    return C_d

def guided_upsampling(C,G,channel,pattern,ep=0):
    r,c=offsets[pattern][channel]
    M=np.zeros_like(C)
    M[r::2,c::2]=1.0
    p=C*M
    return guidedfilter(G,p,M,5,5,ep)

def bilinear_RB(C):
    kernel=np.array([[.25,.5,.25],
                    [.5,1,.5],
                    [.25,.5,.25]], dtype=np.float32)
    return cv2.filter2D(C.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REFLECT_101)    
    
def ri(cfa,pattern):
    cfa_standard = standardize_input(cfa)
    R,G,B=getRGB_known(cfa_standard, pattern)
    
    Grh_lin,Grv_lin=bilinear_1D(G,G,'R',pattern) 
    Rgh_lin,Rgv_lin=bilinear_1D(R,R,'R',pattern)
    Gbh_lin,Gbv_lin=bilinear_1D(G,G,'B',pattern) 
    Bgh_lin,Bgv_lin=bilinear_1D(B,B,'B',pattern)

    Rh_tent=guided_1D(Grh_lin,R,1,'R',pattern)
    Rv_tent=guided_1D(Grv_lin,R,0,'R',pattern)
    Grh_tent=guided_1D(Rgh_lin,G,1,'Gr',pattern)
    Grv_tent=guided_1D(Rgv_lin,G,0,'Gb',pattern)
    Bh_tent=guided_1D(Gbh_lin,B,1,'B',pattern)
    Bv_tent=guided_1D(Gbv_lin,B,0,'B',pattern)
    Gbh_tent=guided_1D(Bgh_lin,G,1,'Gb',pattern)
    Gbv_tent=guided_1D(Bgv_lin,G,0,'Gr',pattern)

    Rh_res=get_residuals(R,Rh_tent,'R',pattern)
    Rv_res=get_residuals(R,Rv_tent,'R',pattern)
    Grh_res=get_residuals(G,Grh_tent,'Gr',pattern)
    Grv_res=get_residuals(G,Grv_tent,'Gb',pattern)
    Bh_res=get_residuals(B,Bh_tent,'B',pattern)
    Bv_res=get_residuals(B,Bv_tent,'B',pattern)
    Gbh_res=get_residuals(G,Gbh_tent,'Gb',pattern)
    Gbv_res=get_residuals(G,Gbv_tent,'Gr',pattern)

    Rh_resin,Rv_resin=bilinear_1D(Rh_res,Rv_res,'R',pattern)
    Grh_resin,Grv_resin=bilinear_1D(Grh_res,Grv_res,'R',pattern)
    Bh_resin,Bv_resin=bilinear_1D(Bh_res,Bv_res,'B',pattern)
    Gbh_resin,Gbv_resin=bilinear_1D(Gbh_res,Gbv_res,'B',pattern)

    Rh=Rh_resin+Rh_tent
    Rv=Rv_resin+Rv_tent
    Grh=Grh_resin+Grh_tent
    Grv=Grv_resin+Grv_tent
    Bh=Bh_resin+Bh_tent
    Bv=Bv_resin+Bv_tent
    Gbh=Gbh_resin+Gbh_tent
    Gbv=Gbv_resin+Gbv_tent

    delta_gr_h=overlap(Grh,'R',pattern)-R+overlap(G,'Gr',pattern)-overlap(Rh,'Gr',pattern)
    delta_gr_v=overlap(Grv,'R',pattern)-R+overlap(G,'Gb',pattern)-overlap(Rv,'Gb',pattern)
    delta_gb_h=overlap(Gbh,'B',pattern)-B+overlap(G,'Gb',pattern)-overlap(Bh,'Gb',pattern)
    delta_gb_v=overlap(Gbv,'B',pattern)-B+overlap(G,'Gr',pattern)-overlap(Bv,'Gr',pattern)

    delta_gr=get_delta(delta_gr_h,delta_gr_v,'R',pattern)
    delta_gb=get_delta(delta_gb_h,delta_gb_v,'B',pattern)

    G_demosaicked=(R+delta_gr)+(B+delta_gb)+G
    G_demosaicked = np.clip(G_demosaicked, 0, 1)

    R_tent=guided_upsampling(R,G_demosaicked,'R',pattern)
    B_tent=guided_upsampling(B,G_demosaicked,'B',pattern)

    R_res=R-overlap(R_tent,'R',pattern)
    B_res=B-overlap(B_tent,'B',pattern)

    R_resin=bilinear_RB(R_res)
    B_resin=bilinear_RB(B_res)

    R_demosaicked=R_resin+R_tent
    R_demosaicked=np.clip(R_demosaicked,0,1)

    B_demosaicked=B_resin+B_tent
    B_demosaicked=np.clip(B_demosaicked,0,1)

    color_image=np.stack((R_demosaicked,G_demosaicked,B_demosaicked),axis=-1)
    
    return color_image

