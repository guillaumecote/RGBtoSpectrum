from matplotlib import pyplot as plt
from matplotlib import patches
#import numpy as np
from scipy.optimize import minimize
import time
from random import randint
import colorsys
import math
import numpy as np

#http://biecoll.ub.uni-bielefeld.de/volltexte/2007/52/pdf/ICVS2007-6.pdf
#'matchingfcn.txt'

def load_matching_fcn():
    with open('matchingfcn.txt','r') as f:
        L=[]
        x=[]
        y=[]
        z=[]
        txt=f.read().split('\n')
        for r in txt[0:len(txt)-1]:
            if not float(r.split('\t')[0]) % 5:
                L.append(float(r.split('\t')[0]))
                x.append(float(r.split('\t')[1]))
                y.append(float(r.split('\t')[2]))
                z.append(float(r.split('\t')[3]))
    return L, x, y, z

def load_lms():
    with open('lms.txt','r') as f:
        Lh=[]
        l=[]
        m=[]
        s=[]
        txt=f.read().split('\n')
        for r in txt[0:len(txt)-1]:
            if not float(r.split('\t')[0]) % 5:
                Lh.append(float(r.split('\t')[0]))
                l.append(float(r.split('\t')[1]))
                m.append(float(r.split('\t')[2]))
                s.append(float(r.split('\t')[3]))
    return Lh, l, m, s

L, x, y, z = load_matching_fcn()
Lh, l, m, s = load_lms()



def plot_matching_fcns(L,x,y,z):
    bottom=[0 for x in range(len(L))]

    indexlist=[]
    for i in range(1,len(x)):
        if (x[i]-y[i])*(x[i-1]-y[i-1])<0 or (x[i]-z[i])*(x[i-1]-z[i-1])<0 or (y[i]-z[i])*(y[i-1]-z[i-1])<0 or (y[i]-z[i])*(y[i-1]-z[i-1])<0:
            indexlist.append(i)

    ind=sorted(indexlist)

    plt.fill_between(L[0:ind[1]],x[0:ind[0]]+y[ind[0]:ind[1]],z[0:ind[1]], facecolor=(0,0,1),alpha=0.25,linewidth=0)
    plt.fill_between(L[ind[1]:ind[3]],z[ind[1]:ind[2]]+x[ind[2]:ind[3]],y[ind[1]:ind[3]], facecolor=(0,1,0),alpha=0.25,linewidth=0)
    plt.fill_between(L[ind[3]:len(L)-1],y[ind[3]:len(L)-1],x[ind[3]:len(L)-1], facecolor=(1,0,0),alpha=0.25,linewidth=0)
    plt.fill_between(L[0:ind[0]],y[0:ind[0]],x[0:ind[0]], facecolor=(1,0,0),alpha=0.25,linewidth=0)
    plt.fill_between(L[0:ind[0]],y[0:ind[0]],x[0:ind[0]], facecolor=(0,0,1),alpha=0.125,linewidth=0)
    plt.fill_between(L[ind[0]:ind[2]],x[ind[0]:ind[2]],y[ind[0]:ind[1]]+z[ind[1]:ind[2]], facecolor=(0,1,0),alpha=0.25,linewidth=0)
    plt.fill_between(L[ind[0]:ind[2]],x[ind[0]:ind[2]],y[ind[0]:ind[1]]+z[ind[1]:ind[2]], facecolor=(0,0,1),alpha=0.125,linewidth=0)
    plt.fill_between(L[ind[2]:len(L)-1],z[ind[2]:len(x)-1],x[ind[2]:ind[3]]+y[ind[3]:len(y)-1], facecolor=(1,0,0),alpha=0.5,linewidth=0)
    plt.fill_between(L[ind[2]:len(L)-1],z[ind[2]:len(x)-1],x[ind[2]:ind[3]]+y[ind[3]:len(y)-1], facecolor=(0,1,0),alpha=0.125,linewidth=0)

    plt.fill_between(L[0:len(L)-1],bottom[0:len(x)-1],y[0:ind[0]]+x[ind[0]:ind[2]]+z[ind[2]:len(L)-1], facecolor=(1,0,0),alpha=0.25,linewidth=0)
    plt.fill_between(L[0:len(L)-1],bottom[0:len(x)-1],y[0:ind[0]]+x[ind[0]:ind[2]]+z[ind[2]:len(L)-1], facecolor=(0,1,0),alpha=0.125,linewidth=0)
    plt.fill_between(L[0:len(L)-1],bottom[0:len(x)-1],y[0:ind[0]]+x[ind[0]:ind[2]]+z[ind[2]:len(L)-1], facecolor=(0,0,1),alpha=0.083,linewidth=0)

# def optfunc(s):
#     df=sum([(s[i+1]-s[i])**2 for i in range(1,len(s)-1)])
#     ddf=sum([(s[i+1]-2*s[i]+s[i-1])**2 for i in range(1,len(s)-1)])
#     return df+ddf

def rgb2xyz(rgb):

    nrgb=rgb
    r=rgb[0]
    g=rgb[1]
    b=rgb[2]

    nrgb=[r/255,g/255,b/255]

    for i in range(3):
       # nrgb[i]=rgb[i]/255
        if (nrgb[i]>0.04045):
            nrgb[i] = ((nrgb[i]+0.055)/1.055)**2.4
        else:
            nrgb[i] = nrgb[i]/12.92
        nrgb[i]=nrgb[i]*100

    xyz=[0,0,0]
    xyz[0] = nrgb[0]*0.4124+nrgb[1]*0.3576+nrgb[2]*0.1805
    xyz[1] = nrgb[0]*0.2126+nrgb[1]*0.7152+nrgb[2]*0.0722
    xyz[2] = nrgb[0]*0.0193+nrgb[1]*0.1192+nrgb[2]*0.9505

    return xyz

def xyz2rgb(xyz):

    xyz=[i/100 for i in xyz]

    R=xyz[0]*3.2406+xyz[1]*-1.5372+xyz[2]*-0.4986
    G=xyz[0]*-0.9689+xyz[1]*1.8758+xyz[2]*0.0415
    B=xyz[0]*0.0557+xyz[1]*-0.2040+xyz[2]*1.0570

    rgb=[R,G,B]

    for i in range(len(rgb)):
        if (rgb[i]>0.0031308):
            rgb[i]=1.055*(rgb[i]**(1/2.4))-0.055
        else:
            rgb[i]=rgb[i]*12.92
        rgb[i]=rgb[i]*255
    min_rgb=min(rgb)
    max_rgb=max(rgb)
    if min_rgb<0 and max_rgb>255:
        rgb=[(x+min_rgb)/(max_rgb)*255 for x in rgb]
    elif min_rgb<0:
        rgb=[max_rgb/(max_rgb-min_rgb)*(x-min_rgb) for x in rgb]
    elif max_rgb>255:
        rgb=[(255-min_rgb)/(max_rgb-min_rgb)*(x-min_rgb)+min_rgb for x in rgb]
    return rgb

def xyz2spectrum(xyz,L,x,y,z):

    dl=L[1]-L[0]
    cons=({'type': 'eq', 'fun':lambda s: sum([dl*s[i]*x[i] for i in range(len(s))])-xyz[0]},
          {'type': 'eq', 'fun':lambda s: sum([dl*s[i]*y[i] for i in range(len(s))])-xyz[1]},
          {'type': 'eq', 'fun':lambda s: sum([dl*s[i]*z[i] for i in range(len(s))])-xyz[2]},
          {'type': 'eq', 'fun':lambda s: s[-1]-s[-2]},
          {'type': 'eq', 'fun':lambda s: s[0]-s[1]})
    optfunc=lambda spect: sum([(spect[i]-spect[i-1])**2 for i in range(1,len(spect))])
    guess=[0 for x in range(len(L))]
    bnds=[[0,float('inf')] for x in range(len(L))]
    res = minimize(optfunc,guess, constraints=cons,method='SLSQP',bounds=bnds,tol=10e-5)
    spectrum=res.x

    return spectrum

def lms2spectrum(lms,L,l,m,s):
    dl=L[1]-L[0]
    cons=({'type': 'eq', 'fun':lambda spect: sum([dl*spect[i]*l[i] for i in range(len(spect))])-lms[0]},
          {'type': 'eq', 'fun':lambda spect: sum([dl*spect[i]*m[i] for i in range(len(spect))])-lms[1]},
          {'type': 'eq', 'fun':lambda spect: sum([dl*spect[i]*s[i] for i in range(len(spect))])-lms[2]})
    optfunc=lambda spect: sum([(spect[i]-spect[i-1])**3 for i in range(1,len(spect))])
    guess=[0 for x in range(len(L))]
    bnds=[[0,float("inf")] for x in range(len(L))]

    res = minimize(optfunc,guess, constraints=cons,method='SLSQP',bounds=bnds)
    spectrum=res.x

    return spectrum

def spectrum2rgb(spectrum,L,x,y,z):

    dl=L[1]-L[0]
    xyz=[0,0,0]
    xyz[0]=sum([dl*x[i]*spectrum[i] for i in range(len(L))])
    xyz[1]=sum([dl*y[i]*spectrum[i] for i in range(len(L))])
    xyz[2]=sum([dl*z[i]*spectrum[i] for i in range(len(L))])

    rgb=xyz2rgb(xyz)

    return rgb

def spectrum2lms(spectrum,L,l,m,s):

    dl=L[1]-L[0]
    lms=[0,0,0]
    lms[0]=sum([dl*l[i]*spectrum[i] for i in range(len(L))])
    lms[1]=sum([dl*m[i]*spectrum[i] for i in range(len(L))])
    lms[2]=sum([dl*s[i]*spectrum[i] for i in range(len(L))])

    return lms

def lms2rgb(lms):
    R=lms[0]*0.0809+lms[1]*-0.1305+lms[2]*0.1167
    G=lms[0]*-0.0102+lms[1]*0.054+lms[2]*-0.1136
    B=lms[0]*-0.0003+lms[1]*-0.0041+lms[2]*0.6935

    return [R,G,B]

def rgb2lms(rgb):
    L=rgb[0]*17.8824+rgb[1]*43.5161+rgb[2]*4.1193
    M=rgb[0]*3.4557+rgb[1]*27.1554+rgb[2]*3.8671
    S=rgb[0]*0.02996+rgb[1]*0.18431+rgb[2]*1.467

    return [L/255,M/255,S/255]



def draw_lms(a, ind, Lh, l, m, s):
   a.fill_between(Lh[0:ind[1]],l[0:ind[0]]+m[ind[0]:ind[1]],s[0:ind[1]], facecolor=(0,0,1),alpha=0.4,linewidth=0)
   a.fill_between(Lh[ind[1]:ind[3]],s[ind[1]:ind[2]]+l[ind[2]:ind[3]],m[ind[1]:ind[3]], facecolor=(0,1,0),alpha=0.4,linewidth=0)
   a.fill_between(Lh[ind[3]:len(Lh)-1],m[ind[3]:len(Lh)-1],l[ind[3]:len(Lh)-1], facecolor=(1,0,0),alpha=0.4,linewidth=0)
   a.fill_between(Lh[0:ind[0]],m[0:ind[0]],l[0:ind[0]], facecolor=(1,0,0),alpha=0.4,linewidth=0)
   a.fill_between(Lh[ind[0]:ind[2]],l[ind[0]:ind[2]],m[ind[0]:ind[1]]+s[ind[1]:ind[2]], facecolor=(0,0.494,0.337),alpha=0.4,linewidth=0)
   a.fill_between(Lh[ind[2]:len(Lh)-1],s[ind[2]:len(l)-1],l[ind[2]:ind[3]]+m[ind[3]:len(l)-1], facecolor=(0.502,0.357,0),alpha=0.4,linewidth=0)
   a.fill_between(Lh[0:len(Lh)-1],bottom[0:len(l)-1],m[0:ind[0]]+l[ind[0]:ind[2]]+s[ind[2]:len(Lh)-1], facecolor=(0.333,0.329,0.224),alpha=0.4,linewidth=0)





fig=plt.figure()

f, axes = plt.subplots(1,3)
f.set_size_inches(18, 4)
lines=[]
titles=['Hue','Saturation','Lightness']
for i, a in enumerate(axes):
   line, = a.plot([], [], 'k-',zorder=1)
   lines.append(line)
   a.set_xlim((390,720))
   a.set_xlabel('Wavelength (nm)')
   a.set_ylabel('Normalized intensity')
   a.set_title(titles[i])

   bottom = [0]* len(Lh)
   lms_cross_list=[]
   for i in range(1,len(l)):
       if (l[i]-m[i])*(l[i-1]-m[i-1])<0 or (l[i]-s[i])*(l[i-1]-s[i-1])<0 or (m[i]-s[i])*(m[i-1]-s[i-1])<0 or (m[i]-s[i])*(m[i-1]-s[i-1])<0:
           lms_cross_list.append(i)

   ind = sorted(lms_cross_list)

   draw_lms(a, ind, Lh, l, m, s)



slist=[[],[],[]]
mlist=[[],[],[]]
reccols=[[],[],[]]
for i in range(25):
   for j in range(3):
       if j==0:
           rgb=colorsys.hls_to_rgb(0.04*i, 0.5,1)
       if j==1:
           rgb=colorsys.hls_to_rgb(0.83, abs(math.cos(i/25*3.14159265)),1)
       if j==2:
           rgb=colorsys.hls_to_rgb(0.83, 0.5,abs(math.cos(i/25*3.14159265)))
       print(i,rgb)
       xyz=rgb2xyz([rgb[0]*255,rgb[1]*255,rgb[2]*255])
       spectrum=xyz2spectrum(xyz,L,x,y,z)
       slist[j].append(spectrum)
       mlist[j].append(max(spectrum))
       reccols[j].append(rgb)

scale=[max(mlist[0]),max(mlist[1]),max(mlist[2])]
for i in range(len(slist[0])):
   for j in range(len(slist)):
       for k in range(len(slist[j][i])):
           slist[j][i][k]=slist[j][i][k]/scale[j]

       rec1=patches.Rectangle((673,0.735),40,0.2,color=(0.3,0.3,0.3),alpha=0.1,zorder=2)
       rec2=patches.Rectangle((670,0.75),40,0.2,color=reccols[j][i],zorder=3)

       axes[j].add_patch(rec1)
       axes[j].add_patch(rec2)
       lines[j].set_data(L,slist[j][i])
#        update_line(i,slist[j][i],lines[j],L)
       plt.savefig('frames/lines'+str(i)+'.png')


############################
############################


#
#    #ims.append((plt.,))
# for spectrumn in slist:
#    plt.plot(L,spectrumn,'k')
#
#
# print(lms2rgb(spectrum2lms(spectrum,L,x,ycb,z)))
# s=spectrum
#
# dl=L[1]-L[0]
# #print(sum([dl*s[i]*x[i] for i in range(len(s))])-lms[0],sum([dl*s[i]*y[i] for i in range(len(s))])-lms[1],sum([dl*s[i]*z[i] for i in range(len(s))])-lms[2])
# for j in range(1):
#
#    #rgb=[randint(0,255),randint(0,255),randint(0,255)]
#    rgb=[150,100,0]
#    lms=rgb2lms(rgb)
#    xyz=rgb2xyz(rgb)
#
#    print('xyz',xyz,'lms',lms)
#    spectrum=xyz2spectrum(xyz,L,x,y,z)
#    maxs=max(spectrum)
#    spectrumn=[s/maxs for s in spectrum]
#    print(spectrum2rgb(spectrum,L,x,y,z))
#    print(lms2rgb(lms))
#    print(spectrum2lms(spectrum,L,l,m,s))
#    nlms=spectrum2lms(spectrum,L,l,m,s)
#    out=lms2rgb(nlms)
#    sumrgb=sum(rgb)
# #    nout=[i/sum(out)*sumrgb for i in out]
#
#
#
# #
#    plt.plot(L,x,'r',L,y,'g',L,z,'b',L,l,'r',L,m,'g',L,s,'b',L,spectrumn,'k')
#    plt.xlim([380,750])
#    plt.show()
#    print('lms',lms)
#    print(sum([dl*spectrum[i]*x[i] for i in range(len(spectrum))])-xyz[0],sum([dl*spectrum[i]*y[i] for i in range(len(spectrum))])-xyz[1],sum([dl*spectrum[i]*z[i] for i in range(len(spectrum))])-xyz[2])
#    print(sum([dl*spectrum[i]*l[i] for i in range(len(spectrum))])-lms[0],sum([dl*spectrum[i]*m[i] for i in range(len(spectrum))])-lms[1],sum([dl*spectrum[i]*s[i] for i in range(len(spectrum))])-lms[2])
#    print(sum([dl*spectrum[i]*l[i] for i in range(len(spectrum))]),sum([dl*spectrum[i]*m[i] for i in range(len(spectrum))]),sum([dl*spectrum[i]*s[i] for i in range(len(spectrum))]))
#    print(rgb,out,[out[i]-rgb[i] for i in range(3)])
#
#    plt.figure(1)
#    ax1=plt.subplot(211)
#    rec1=patches.Rectangle((0,0),1,1,color=[i/255 for i in rgb])
#    ax1.add_patch(rec1)
#    ax2=plt.subplot(212)
#    rec2=patches.Rectangle((0,0),1,1,color=[i/255 for i in out])
#    ax2.add_patch(rec2)
#    plt.show()
#
#
#
# toc=time.clock()
#
# print(toc-tic)
# spectrum=rgb2spectrum([255,0,255],L,x,y,z)
#
#
#
#
# plt.plot(L,x,L,y,L,z)
# plt.xlim((380,750))
# plt.show()
# plt.plot(L,x,L,ycb,L,z)
# plt.xlim((380,750))
# plt.show()
#
# fx=[x[i]*filt[i] for i in range(len(L))]
# fy=[y[i]*filt[i] for i in range(len(L))]
# fycb=[ycb[i]*filt[i] for i in range(len(L))]
# fz=[z[i]*filt[i] for i in range(len(L))]
#
# plt.plot(L,fx,L,fy,L,fz)
# plt.xlim((380,750))
# plt.show()
# plt.plot(L,fx,L,fycb,L,fz,L,filt)
# plt.xlim((380,750))
# plt.show()
# plot_matching_fcns(L,x,y,z)
# plot_matching_fcns(L,x,ycb,z)
