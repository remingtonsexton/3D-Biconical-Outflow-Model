# AGN Biconical Outflow Model adopted from Bae & Woo 2016
# See: https://ui.adsabs.harvard.edu/abs/2016ApJ...828...97B/abstract
# 

import numpy as np
import pandas as pd
import scipy as sp
# import emcee
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.spatial import Delaunay
from matplotlib.patches import FancyArrowPatch
from matplotlib import animation
from scipy.interpolate import griddata, interp1d
from scipy.ndimage import gaussian_filter
from scipy.integrate import simps
from astropy.convolution import convolve
from astropy.convolution.kernels import Gaussian2DKernel
import matplotlib.cm as cm
import os 
import shutil
import sys
import matplotlib
import time
import emcee
matplotlib.rcParams['agg.path.chunksize'] = 100000
# np.set_printoptions(threshold=sys.maxsize)

# plt.style.use('dark_background')

#####################################################################################

def bicone_inclination(B1_deg,B2_deg):
    # Convert angles to radians
    # NOTE: PA_bicone = 0.0 => theta_B2 = 0.0 (bicone oriented N-S; no rotation along line of sight)
    B1_rad = B1_deg * np.pi/180.0
    B2_rad = B2_deg * np.pi/180.0
    if (np.sin(B1_rad)*np.cos(B1_rad)>=0.0): S = 1.0
    elif (np.sin(B1_rad)*np.cos(B1_rad)<0.0): S = -1.0
    i_bicone = S*np.arccos(np.sqrt(np.sin(B2_rad)**2 + (np.cos(B1_rad)*np.cos(B2_rad))**2))
    i_bicone_deg = i_bicone * 180.0/np.pi
    return i_bicone_deg

#####################################################################################

def dust_inclination(D1_deg,D3_deg=0.0):
    # Convert angles to radian
    # NOTE: PA_dust = 0.0 => theta_D3 = 0.0 (dust plane oriented N-S; no tilt along line of sight)
    D1_rad = D1_deg * np.pi/180.0
    D3_rad = D3_deg * np.pi/180.0
    if (np.sin(D1_rad)*np.cos(D1_rad)>=0.0): S = 1.0
    elif (np.sin(D1_rad)*np.cos(D1_rad)<0.0): S = -1.0
    i_dust = S*np.arccos( np.cos(D1_rad)*np.cos(D3_rad) )
    i_dust_deg = i_dust * 180.0/np.pi
    return i_dust

#####################################################################################

def generate_bicone(theta_in_deg, theta_out_deg,
                    theta_B1_deg, theta_B2_deg, theta_B3_deg,
                    theta_D1_deg, theta_D2_deg, theta_D3_deg,
                    D=1.0, tau=5.0, fn=1.0, A=0.90,
                    vmax=1000.0, vtype='decreasing',
                    sampling=100,plot=True,orientation=(15,45),save_fig=True):
    #################################################################################
    # generate_bicone() 
    # 
    # DESCRIPTION:
    # -----------
    #         This function generates the bicone coordinate, flux, and velocity
    #         grids needed to construct 2D maps and LOSVD profiles.  The function
    #         calculates the flux and velocity profiles on a uniform grid, determines
    #         flux and velocity profiles, performs coordinate rotations of both the 
    #         bicone and dust planes, performs grid interpolation back onto a 
    #         Cartesian grid, calculates the LOS velocity for the velocity 
    #         grid, and outputs the grids.
    # 
    # INPUTS:
    # ------
    #         theta_in_deg  : inner bicone angle measured from reflected on z-axis
    #         theta_out_deg : outer bicone angle measured from reflected on z-axis
    #         theta_B1_deg  : ccw rotation of bicone about x-axis in degrees
    #         theta_B2_deg  : ccw rotation of bicone about y-axis in degrees
    #         theta_B3_deg  : ccw rotation of bicone about z-axis in degrees
    #         theta_D1_Deg  : ccw rotation of dust plane about x-axis in degrees  
    #         theta_D2_Deg  : ccw rotation of dust plane about y-axis in degrees
    #         theta_D3_Deg  : ccw rotation of dust plane about z-axis in degrees
    #         D             : length of bicone in arbitary units
    #         tau           : flux profile parameter; e.g. tau=5 means flux decreases 
    #                         by 1/150 from value at nucleus
    #         fn            : flux value at nucleus in arbitrary units
    #         sampling      : grid sampling factor; e.g. sampling = 5 generates
    #                         a grid of 5 x 5 x 5 = 125 (x,y,z) points
    #
    # OUTPUTS:
    # -------
    #         fgrid : an (N,3) grid of flux values at each location on the grid
    #         vgrid : an (N,3) grid of LOS velocities at each location on the grid

    #         
    # NOTES:
    # ----- 
    #         

    #################################################################################
    if int(sampling)%2==0:
        # Having an odd sampling ensures that there is a value at (0,0,0)
        sampling = int(sampling)+1
    # print('\n Bicone grid sampling = %d' % sampling)
    # Generate a uniform meshgrid in Cartesian coordinates 
    # print('\n Generating bicone...')
    Xg, Yg, Zg = np.meshgrid(np.linspace(-D,D,sampling), np.linspace(-D,D,sampling), np.linspace(-D,D,sampling) )
    xbgrid, ybgrid, zbgrid = Xg.ravel(), Yg.ravel(), Zg.ravel()
    # Calculate d
    d = (xbgrid**2+ybgrid**2+zbgrid**2)**0.5
    ind = np.where(d<=D)[0] # indices within a spere of radius D
    # Generate grid of zeros which we will occupy for valid values of the bicone
    # 1 for valid entries, 0 for none; this will be used for flux and velocity grids
    bicone_grid = np.zeros(len(d))
    # Calculate r_in,r_out
    theta_in_rad  = theta_in_deg * np.pi/180.
    theta_out_rad = theta_out_deg * np.pi/180.
    r_in  = (xbgrid**2+ybgrid**2)**0.5/np.cos(np.pi/2.0 - theta_in_rad)
    r_out = (xbgrid**2+ybgrid**2)**0.5/np.cos(np.pi/2.0 - theta_out_rad)
    # Calculate min and max z at (x,y)
    z_min = r_out * np.cos(theta_out_rad)
    z_max = r_in * np.cos(theta_in_rad)
    # For all entries with (d<=D), if the absolute value of a z value of the grid
    # lies between the minimum and maximum z, that point on the grid is a valid entry.
    # t0 = time.time()
    # for i in ind:
    #     if (z_min[i]<=np.abs(zbgrid[i])<=z_max[i]):
    #         bicone_grid[i]=1.0

    ind = np.where((d<=D) & (np.abs(zbgrid)<=z_max) & (np.abs(zbgrid)>=z_min))
    bicone_grid[ind]=1.0
    # print('\n      %0.2f seconds' % float(time.time()-t0))
    # Perform coordinate rotation of bicone
    # print('\n      Performing bicone coordinate rotation...')
    # t0 = time.time()
    R = coord_rotation((theta_B1_deg,theta_B2_deg,theta_B3_deg))
    u = np.vstack((xbgrid,ybgrid,zbgrid))
    u_rot = np.dot(R,u)
    xb_rot,yb_rot,zb_rot = u_rot[0],u_rot[1],u_rot[2]
    # print('\n      %0.2f seconds' % float(time.time()-t0))
    # Interpolate onto a new Cartesian grid of the same dimension
    # t0 = time.time()


    # print('\n      Performing bicone interpolation back onto normal grid...')

    # scipy.interpolate.griddata uses Delauney triangulation for irregular grids which 
    # makes this method very slow.
    points = zip(xb_rot,yb_rot,zb_rot) # 
    values = bicone_grid # 

    new_bicone_grid = griddata(np.array(points),np.array(values),np.array(zip(xbgrid,ybgrid,zbgrid)),method='nearest')#,fill_value=0) # works version 1.1.0
    # new_bicone_grid = griddata(np.array((xb_rot,yb_rot,zb_rot)).T,np.array(values),np.array((xbgrid,ybgrid,zbgrid)).T,method='nearest')#,fill_value=0) # 

    # print('\n      %0.2f seconds' % float(time.time()-t0))



    # Generate dust plane grid
    # print('\n Generating dust plane...')
    rd = 2.0*D # dust plane radius (defualt 2 x D)
    Xg, Yg = np.meshgrid(np.linspace(-rd,rd,sampling),np.linspace(-rd,rd,sampling) )
    xdgrid,ydgrid = Xg.ravel(),Yg.ravel()
    zdgrid = np.full(np.shape(xdgrid),0.0) # zero thickness for simplicity
    d = (xdgrid**2+ydgrid**2)**0.5
    xdg, ydg, zdg = xdgrid[d<=rd], ydgrid[d<=rd], zdgrid[d<=rd]
    # Rotate the dust plane in 3d
    # print('\n      Performing dust plane coordinate rotation...')
    u = np.vstack((xdg, ydg, zdg))
    R = coord_rotation((theta_D1_deg,theta_D2_deg,theta_D3_deg))
    u_rot = np.dot(R,u)
    xd_rot,yd_rot,zd_rot = u_rot[0],u_rot[1],u_rot[2]

    bicone_coords = (xbgrid, ybgrid, zbgrid)
    dust_coords = (xd_rot,yd_rot,zd_rot)
    # Generate flux grid
    fgrid = flux_profile(new_bicone_grid,bicone_coords,dust_coords,tau=tau,D=D,fn=fn,A=A)
    vgrid = velocity_profile(new_bicone_grid,bicone_coords,D=D,vmax=vmax,vtype=vtype)
    
    # Plot model of the bicone and dust plane in 3d for visualization purposes
    if plot==True:
        bicone_vec = bicone_vector(theta_B1_deg,theta_B2_deg,theta_B3_deg)
        dust_vec   = dust_vector(theta_D1_deg,theta_D2_deg,theta_D3_deg)
        plot_model(bicone_coords,dust_coords,fgrid,vgrid,bicone_vec,dust_vec,save_fig=save_fig,orientation=orientation)
        # plot_model_3D(bicone_coords,dust_coords,fgrid,vgrid,bicone_vec,dust_vec,save_fig=save_fig,orientation=orientation)



    return xbgrid,ybgrid,zbgrid,fgrid,vgrid

#####################################################################################

def velocity_profile(bicone_grid,bicone_coords,D=1.0,vmax=1000.0,vtype='decreasing'):
    # Calculates the velocity at a distance d from the nucleus
    xb,yb,zb = bicone_coords
    d = (xb**2+yb**2+zb**2)**0.5
    if (vtype=='increasing') or (vtype==1):
        # Linearly increasing
        # vd = 0 at d =0, vd = vmax at d=D=1
        k = float(vmax)/float(D)
        vd = k*d
    elif (vtype=='decreasing') or (vtype==2):
        # Linearly decreasing
        # vd = vmax at d = 0, vd = 0 at d=D=1
        k = float(vmax)/float(D)
        vd = vmax-k*d
        
    elif (vtype=='constant') or (vtype==3):
        # Constant velocity at all d
        vd = float(vmax)

    # Calculate velocity on the bicone grid
    vgrid = bicone_grid * vd

    # Calculate the projected velocity along the LOS
    cos_i = (-yb/(yb**2+zb**2)**0.5)
    cos_i[~np.isfinite(cos_i)]=0
    vp = vgrid  * cos_i

    return vp

#####################################################################################

def flux_profile(bicone_grid,bicone_coords,dust_coords,tau=5.0,D=1.0,fn=1.0,A=0.90):
    # print('\n Generating flux profile...\n')
    xb,yb,zb = bicone_coords
    xd,yd,zd = dust_coords
    d = ( (xb)**2 + (yb)**2 + (zb)**2 )**0.5
    
    points = (xd,zd)
    values = yd
    # Get the y-coordinate of the dust plane at the locations 
    # of the bicone at the x-z coordinates
    yc = griddata(points, values, (xb, zb), method='linear')

    fd_ext = fn*np.exp(-tau*(d/D))
    fd_ext = fd_ext*bicone_grid
    # t0 = time.time()
    # for i in range(0,len(fd_ext),1):
    #     if yb[i]<(yc[i]):
    #         # If dust plane is in front of bicone, extinguish the flux by factor of 1-A
    #         fd_ext[i]=fd_ext[i]*(1.0-A)
    # print('\n      %0.2f seconds' % float(time.time()-t0))

    # t0 = time.time()
    ind =  np.where((yb<=yc))[0]
    fd_ext[ind]*=(1.0-A)
    # print('\n      %0.2f seconds' % float(time.time()-t0))
    flux = fd_ext

    return np.array(flux)

#####################################################################################

def coord_rotation(theta):
    # Convert to radians
    if theta[0]==0.0: theta = [0.001,theta[1],theta[2]]
    if theta[1]==0.0: theta = [theta[0],0.001,theta[2]]
    if theta[2]==0.0: theta = [theta[0],theta[1],0.001]
        
    if theta[0]==90.0: theta = [89.9,theta[1],theta[2]]
    if theta[1]==90.0: theta = [theta[0],89.9,theta[2]]
    if theta[2]==90.0: theta = [theta[0],theta[1],89.9]

    if theta[0]==-90.0: theta = [-89.9,theta[1],theta[2]]
    if theta[1]==-90.0: theta = [theta[0],-89.9,theta[2]]
    if theta[2]==-90.0: theta = [theta[0],theta[1],-89.9]
    
    theta_1_rad = theta[0] * np.pi/180.0
    theta_2_rad = theta[1] * np.pi/180.0
    theta_3_rad = theta[2] * np.pi/180.0
    # The bicone and dust angles correspond to Euler angles which are 
    # (e1,e2,e3) -> (rotation about z, rotation about x, rotation about z again)
    theta_1, theta_2, theta_3 = theta_1_rad, theta_2_rad, theta_3_rad
    R_x = np.array([[1,         0,                  0                 ],
                    [0,         np.cos(theta_1),   -np.sin(theta_1)   ],
                    [0,         np.sin(theta_1),    np.cos(theta_1)   ]
                    ])
    R_y = np.array([[ np.cos(theta_2),     0,        np.sin(theta_2)    ],
                    [ 0,                   1,        0                  ],
                    [-np.sin(theta_2),     0,        np.cos(theta_2)    ]
                    ])
    R_z = np.array([[np.cos(theta_3),       -np.sin(theta_3),        0],
                    [np.sin(theta_3),        np.cos(theta_3),        0],
                    [0,                      0,                      1]
                    ])             
    R  = np.dot(R_z, np.dot( R_y, R_x )) #np.dot(R_z, np.dot( R_y, R_x ))
#     RR = np.dot(R_x, np.dot( R_y, R_z ))
    return R

#####################################################################################

def bicone_vector(theta_B1_deg,theta_B2_deg,theta_B3_deg):
    R = coord_rotation((theta_B1_deg,theta_B2_deg,theta_B3_deg))
    c1 = np.array([0.0,0.0,0.0])
    c2 = np.array([0.0,0.0,2.0])
    c1_rot = c1
    u = np.vstack((c2[0],c2[1],c2[2]))
    c2_rot = np.dot(R,u)
    bicone_vec = [[c1_rot[0],c2_rot[0][0]], [c1_rot[1], c2_rot[1][0]], [c1_rot[2], c2_rot[2][0]]]

    return bicone_vec

#####################################################################################

def dust_vector(theta_D1_deg,theta_D2_deg,theta_D3_deg):
    R = coord_rotation((theta_D1_deg,theta_D2_deg,theta_D3_deg))
    c1 = np.array([0.0,0.0,0.0])
    c2 = np.array([0.0,0.0,2.5])
    c1_rot = c1
    u = np.vstack((c2[0],c2[1],c2[2]))
    c2_rot = np.dot(R,u)
    dust_vec = [[c1_rot[0],c2_rot[0][0]], [c1_rot[1], c2_rot[1][0]], [c1_rot[2], c2_rot[2][0]]]
    return dust_vec

#Custom class for 3D arrows
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
 
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
 
    def set_data(self, xs, ys, zs):
        self._verts3d = xs, ys, zs

#####################################################################################

def map_2d(xb, yb, zb, fgrid, vgrid, D=1.0, sampling=100, interpolation='none',plot=True,save_fig=True):
    if int(sampling)%2==0:
        # Having an odd sampling ensures that there is a value at (0,0,0)
        sampling = int(sampling)+1 
    # Reshape grids into cubes
    fgrid = fgrid.reshape(sampling,sampling,sampling)
    vgrid = vgrid.reshape(sampling,sampling,sampling)
    #### Flux map ###
    fmap = simps(fgrid,axis=0)
    fmap[fmap<=0]=1
    fmap = np.log10(fmap)
    fmap[fmap<=0] = np.nan
    fmap = fmap + np.abs(np.nanmin(fmap))
    
    #### Velocity map ###

    vmap = simps(np.multiply(fgrid,vgrid),axis=0)/simps(fgrid,axis=0)


    ### Dispersion map ###

    dmap = ( simps(np.multiply(fgrid,vgrid**2),axis=0)/simps(fgrid,axis=0) - vmap**2 )**0.5

    # Integrated velocity along LOS
    F = simps(simps(simps(fgrid,axis=2),axis=1),axis=0)
    v_int = simps(simps(simps(vgrid*fgrid,axis=2),axis=1),axis=0)/F
    d_int = ( simps(simps(simps(vgrid**2*fgrid,axis=2),axis=1),axis=0)/F - v_int**2 )**0.5

    print('Integrated velocity = %s (km/s)' % v_int)
    print('Integrated velocity dispersion = %s (km/s)' % d_int)

    # Plot
    if plot==True:

        fig = plt.figure(figsize=(14,5))
        ax1 = fig.add_subplot(1,3,1)
        ax2 = fig.add_subplot(1,3,2)
        ax3 = fig.add_subplot(1,3,3)

        # Plot flux map

        # Axes 1: Flux Map

        # Truncate colormap
        def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
            new_cmap = colors.LinearSegmentedColormap.from_list(
                'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                cmap(np.linspace(minval, maxval, n)))
            return new_cmap
        cmap = plt.get_cmap('nipy_spectral')
        new_cmap = truncate_colormap(cmap, 0.0, 0.9)


        flux_axes = ax1.imshow(fmap.T,cmap=new_cmap,interpolation=interpolation,origin='lower',
                   vmin=(np.nanmin(fmap)),vmax=(np.nanmax(fmap)),
                   extent=[np.min(xb),np.max(xb),np.min(zb),np.max(zb)])

        # Number of levels for contours
        n = np.copy(fmap)
        n[fmap/fmap!=1]=0
        nlevels = len(np.unique(np.round(n))) * 2

        ax1.contour(fmap.T,extent=[np.min(xb),np.max(xb),np.min(zb),np.max(zb)],colors='black',linewidths=0.5,alpha=0.75,
                    levels=np.linspace((np.nanmin(fmap)),(np.nanmax(fmap)+1),nlevels))
        
        ax1.set_xlim(-1.1,1.1)
        ax1.set_ylim(-1.1,1.5)
        cbax1 = inset_axes(ax1, width="90%", height="5%", loc=9)
        fig.colorbar(flux_axes, cax=cbax1, orientation='horizontal')
        ax1.set_title(r'$\log_{10}$ Flux',fontsize=16)
        ax1.set_ylabel(r'projected distance ($d/D$)',fontsize=12)
        ax1.xaxis.set_tick_params(labelsize=12)
        ax1.yaxis.set_tick_params(labelsize=12)
        ax1.invert_xaxis()

        # Axes 2: Velocity Map
        vel_axes = ax2.imshow(vmap.T,
                   extent=[np.min(xb),np.max(xb),np.min(zb),np.max(zb)],cmap=cm.RdBu_r,
                   vmin=-(np.nanmax(np.abs(vmap))),vmax=(np.nanmax(np.abs(vmap))),interpolation=interpolation,origin='lower')
        ax2.contour(fmap.T,extent=[np.min(xb),np.max(xb),np.min(zb),np.max(zb)],colors='black',linewidths=0.5,alpha=0.75,
                    levels=np.linspace(np.nanmin(fmap),np.nanmax(fmap)+1,nlevels))


        ax2.set_xlim(-1.1,1.1)
        ax2.set_ylim(-1.1,1.5)
        cbax2 = inset_axes(ax2, width="90%", height="5%", loc=9)
        fig.colorbar(vel_axes, cax=cbax2, orientation='horizontal')
        ax2.set_title(r'Velocity (km s$^{-1}$)',fontsize=16)
        ax2.set_xlabel(r'projected distance ($d/D$)',fontsize=12)
        ax2.xaxis.set_tick_params(labelsize=12)
        ax2.yaxis.set_tick_params(labelsize=12)
        ax2.invert_xaxis()


        # Axes 3: Velocity Dispersion Map
        disp_axes = ax3.imshow(dmap.T,
                   extent=[np.min(xb),np.max(xb),np.min(zb),np.max(zb)],cmap=cm.Blues,
                   vmin=(np.nanmin(dmap)),vmax=(np.nanmax(dmap)),interpolation=interpolation,origin='lower')
        ax3.contour(fmap.T,extent=[np.min(xb),np.max(xb),np.min(zb),np.max(zb)],colors='black',linewidths=0.5,alpha=0.75,
                    levels=np.linspace(np.nanmin(fmap),np.nanmax(fmap)+1,nlevels))

        ax3.set_xlim(-1.1,1.1)
        ax3.set_ylim(-1.1,1.5)
        cbax3 = inset_axes(ax3, width="90%", height="5%", loc=9)
        fig.colorbar(disp_axes, cax=cbax3, orientation='horizontal')
        ax3.set_title(r'Velocity Dispersion (km s$^{-1}$)',fontsize=16)
        ax3.xaxis.set_tick_params(labelsize=12)
        ax3.yaxis.set_tick_params(labelsize=12)
        ax3.invert_xaxis()
        
        plt.tight_layout()
        if save_fig==True:
            # plt.savefig('maps_2d.pdf',dpi=150,fmt='pdf')
            plt.savefig('maps_2d.png',dpi=300,fmt='png')

        # plt.close()


    return fmap.T, vmap.T, dmap.T, v_int, d_int


#####################################################################################

def plot_model(bicone_coords,dust_coords,flux_profile,vel_profile,bicone_vec,dust_vec,save_fig,orientation):

    azim, elev = orientation

    xb,yb,zb = bicone_coords
    xd,yd,zd = dust_coords
    fd = flux_profile
    vd = np.copy(vel_profile)
    vd[vd==0] = np.nan

    points = (xd,yd)
    values = zd
    # Put on uniform grid
    xdgrid,ydgrid = np.meshgrid(np.linspace(np.min(xd),np.max(xd),1000),np.linspace(np.min(yd),np.max(yd),1000))
    zdgrid = griddata(points, values, (xdgrid, ydgrid), method='cubic')

    # Truncate flux colormap
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap
    cmap = plt.get_cmap('nipy_spectral')
    new_cmap = truncate_colormap(cmap, 0.0, 0.9)

    fig = plt.figure(figsize=(14,10))
    gs = gridspec.GridSpec(ncols=9, nrows=9, figure=fig)
    ax1 = fig.add_subplot(gs[0:4, 0:3])
    ax2 = fig.add_subplot(gs[0:4, 3:6])
    ax3 = fig.add_subplot(gs[0:4, 6:9])
    ax4 = fig.add_subplot(gs[4:9, 0:4],projection='3d')
    ax5 = fig.add_subplot(gs[4:9, 5:9],projection='3d')
    # Plotting function for animation
    fontsize = 12
    axis_size = 2
    alpha = 1.0
    #################################################################################
    # Axis 1: x-y projection
    ax1.scatter(xb,yb,c=np.log10(fd),alpha=0.25,marker='.',s=1,zorder=10, cmap=new_cmap)
    ax1.contourf(xdgrid, ydgrid, zdgrid, cmap=cm.Oranges,alpha=0.5)
    ax1.arrow(bicone_vec[0][0], bicone_vec[1][0], bicone_vec[0][1], bicone_vec[1][1],
              width=0.05, head_width=0.2, head_length=0.2,length_includes_head=True,
              color='xkcd:cerulean',alpha=alpha,zorder=15)
    ax1.arrow(dust_vec[0][0], dust_vec[1][0], dust_vec[0][1], dust_vec[1][1],
              width=0.05, head_width=0.2, head_length=0.2,length_includes_head=True,
              color='xkcd:dark orange',alpha=alpha,zorder=12)
    ax1.arrow(0, 2.5, 0, -1,
              width=0.05, head_width=0.2, head_length=0.2,length_includes_head=True,
              color='xkcd:red',alpha=alpha,zorder=12)
    ax1.set_xlim(-2.5,2.5)
    ax1.set_ylim(-2.5,2.5)
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$y$')
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    ax1.grid(color='black',alpha=0.5)
    ax1.set_title('Looking down')
    #################################################################################
    # Axis 2: y-z projection
    ax2.scatter(yb,zb,c=np.log10(fd),alpha=0.25,marker='.',s=1,zorder=10, cmap=new_cmap)
    ax2.contourf(ydgrid,zdgrid,zdgrid, cmap=cm.Oranges,alpha=0.5)
    ax2.arrow(bicone_vec[1][0], bicone_vec[2][0], bicone_vec[1][1], bicone_vec[2][1],
              width=0.05, head_width=0.2, head_length=0.2,length_includes_head=True,
              color='xkcd:cerulean',alpha=alpha,zorder=15)
    ax2.arrow(dust_vec[1][0], dust_vec[2][0], dust_vec[1][1], dust_vec[2][1],
              width=0.05, head_width=0.2, head_length=0.2,length_includes_head=True,
              color='xkcd:dark orange',alpha=alpha,zorder=12)
    ax2.arrow(2.5, 0, -1, 0,
              width=0.05, head_width=0.2, head_length=0.2,length_includes_head=True,
              color='xkcd:red',alpha=alpha,zorder=12)
    ax2.set_xlim(-2.5,2.5)
    ax2.set_ylim(-2.5,2.5)
    ax2.set_xlabel(r'$y$')
    ax2.set_ylabel(r'$z$')
    ax2.grid(color='black',alpha=0.5)
    ax2.set_title('Side view')
    #################################################################################
    # Axis 3: x-z (LOS) projection
    ax3.scatter(xb,zb,c=np.log10(fd),alpha=0.25,marker='.',s=1,zorder=10, cmap=new_cmap)
    ax3.contourf(xdgrid, zdgrid,zdgrid, cmap=cm.Oranges,alpha=0.5)
    ax3.arrow(bicone_vec[0][0], bicone_vec[2][0], bicone_vec[0][1], bicone_vec[2][1],
              width=0.05, head_width=0.2, head_length=0.2,length_includes_head=True,
              color='xkcd:cerulean',alpha=alpha,zorder=15)
    ax3.arrow(dust_vec[0][0], dust_vec[2][0], dust_vec[0][1], dust_vec[2][1],
              width=0.05, head_width=0.2, head_length=0.2,length_includes_head=True,
              color='xkcd:dark orange',alpha=alpha,zorder=12)
    ax3.plot(0, 0,color='xkcd:red',alpha=alpha,marker='o',zorder=12)
    ax3.set_xlim(-2.5,2.5)
    ax3.set_ylim(-2.5,2.5)
    ax3.set_xlabel(r'$y$')
    ax3.set_ylabel(r'$z$')
    ax3.invert_xaxis()
    ax3.grid(color='black',alpha=0.5)
    ax3.set_title('Along L.O.S.')
    #################################################################################
    # Axis 4: 3D Flux Plot
    # Bicone
    flux = ax4.scatter(xb,yb,zb, marker='.',s=1,c=np.log10(fd),  alpha=0.25,zorder=10,cmap=new_cmap)
    B_arrow = Arrow3D(bicone_vec[0], bicone_vec[1], bicone_vec[2], arrowstyle="-|>", lw=3,mutation_scale=20,color='xkcd:cerulean blue',alpha=0.5)
    ax4.add_artist(B_arrow)
    # Dust Plane
    ax4.plot_wireframe(xdgrid,ydgrid,zdgrid,alpha=0.25,color='xkcd:orange',zorder=3)
    D_arrow = Arrow3D(dust_vec[0], dust_vec[1], dust_vec[2], arrowstyle="-|>", lw=3,mutation_scale=20,color='xkcd:dark orange',alpha=0.5)
    ax4.add_artist(D_arrow)
    # Axes labels annd lines
    LOS_arrow = Arrow3D([0,0], [2,1.], [0,0], arrowstyle="-|>", lw=3,mutation_scale=20,color='xkcd:red',alpha=alpha)
    ax4.add_artist(LOS_arrow)
    ax4.text(0, 2, +0.1, "L.O.S.", color='black',size=fontsize,alpha=alpha)
    # x-axis
    ax4.set_xlim(-2,2)
    ax4.set_xlabel(r'$x$',fontsize=fontsize)
    xAxisLine_pos = Arrow3D([0,axis_size], [0,0], [0,0], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    xAxisLine_neg = Arrow3D([0,-axis_size], [0,0], [0,0], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    ax4.add_artist(xAxisLine_pos)
    ax4.add_artist(xAxisLine_neg)
    ax4.text(axis_size+0.1, 0, -0.1, r'$x$', color='black',size=fontsize,alpha=alpha)
    # y-axis
    ax4.set_ylim(-2,2)
    ax4.set_ylabel(r'$y$',fontsize=fontsize)
    yAxisLine_pos = Arrow3D([0,0], [0,axis_size], [0,0], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    yAxisLine_neg = Arrow3D([0,0], [0,-axis_size], [0,0], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    ax4.add_artist(yAxisLine_pos)
    ax4.add_artist(yAxisLine_neg)
    ax4.text(0, axis_size+0.1, -0.1, r'$y$', color='black',size=fontsize,alpha=alpha)
    # z-axis
    ax4.set_zlim(-2,2)
    ax4.set_zlabel(r'$z$',fontsize=fontsize)
    zAxisLine_pos = Arrow3D([0,0], [0,0], [0,axis_size], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    zAxisLine_neg = Arrow3D([0,0], [0,0], [0,-axis_size], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    ax4.add_artist(zAxisLine_pos)
    ax4.add_artist(zAxisLine_neg)
    ax4.text(0, 0, axis_size+0.1, r'$z$', color='black',size=fontsize,alpha=alpha)
    # Add a color bar which maps values to colors.
    fig.colorbar(flux, ax=ax4, shrink=0.5, aspect=5,label=r'$\log_{10}$ Flux')
    ax4.view_init(azim=azim,elev=elev)

    #################################################################################
    # Axis 5: 3D Velocity Plot
    # Bicone
    vel = ax5.scatter(xb,yb,zb, marker='.',s=1,c=vd, cmap=cm.RdBu_r, alpha=0.25,zorder=10)
    B_arrow = Arrow3D(bicone_vec[0], bicone_vec[1], bicone_vec[2], arrowstyle="-|>", lw=3,mutation_scale=20,color='xkcd:cerulean blue',alpha=0.5)
    ax5.add_artist(B_arrow)
    # Dust Plane
    ax5.plot_wireframe(xdgrid,ydgrid,zdgrid,alpha=0.25,color='xkcd:orange',zorder=3)
    D_arrow = Arrow3D(dust_vec[0], dust_vec[1], dust_vec[2], arrowstyle="-|>", lw=3,mutation_scale=20,color='xkcd:dark orange',alpha=0.5)
    ax5.add_artist(D_arrow)
    # Axes labels annd lines
    LOS_arrow = Arrow3D([0,0], [2,1.], [0,0], arrowstyle="-|>", lw=3,mutation_scale=20,color='xkcd:red',alpha=alpha)
    ax5.add_artist(LOS_arrow)
    ax5.text(0, 2, +0.1, "L.O.S.", color='black',size=fontsize,alpha=alpha)
    # x-axis
    ax5.set_xlim(-2,2)
    ax5.set_xlabel(r'$x$',fontsize=fontsize)
    xAxisLine_pos = Arrow3D([0,axis_size], [0,0], [0,0], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    xAxisLine_neg = Arrow3D([0,-axis_size], [0,0], [0,0], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    ax5.add_artist(xAxisLine_pos)
    ax5.add_artist(xAxisLine_neg)
    ax5.text(axis_size+0.1, 0, -0.1, r'$x$', color='black',size=fontsize,alpha=alpha)
    # y-axis
    ax5.set_ylim(-2,2)
    ax5.set_ylabel(r'$y$',fontsize=fontsize)
    yAxisLine_pos = Arrow3D([0,0], [0,axis_size], [0,0], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    yAxisLine_neg = Arrow3D([0,0], [0,-axis_size], [0,0], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    ax5.add_artist(yAxisLine_pos)
    ax5.add_artist(yAxisLine_neg)
    ax5.text(0, axis_size+0.1, -0.1, r'$y$', color='black',size=fontsize,alpha=alpha)
    # z-axis
    ax5.set_zlim(-2,2)
    ax5.set_zlabel(r'$z$',fontsize=fontsize)
    zAxisLine_pos = Arrow3D([0,0], [0,0], [0,axis_size], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    zAxisLine_neg = Arrow3D([0,0], [0,0], [0,-axis_size], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    ax5.add_artist(zAxisLine_pos)
    ax5.add_artist(zAxisLine_neg)
    ax5.text(0, 0, axis_size+0.1, r'$z$', color='black',size=fontsize,alpha=alpha)
    # Add a color bar which maps values to colors.
    fig.colorbar(vel, ax=ax5, shrink=0.5, aspect=5,label=r'Projected Velocity along LOS (km/s)')
    ax5.view_init(azim=azim,elev=elev)

    plt.tight_layout()
    gs.tight_layout(fig)

    if save_fig==True:
        # plt.savefig('model_3d.pdf',dpi=150,fmt='pdf')
        plt.savefig('model_3d.png',dpi=300,fmt='png')
    # plt.close()
    return None

def plot_model_3D(bicone_coords,dust_coords,flux_profile,vel_profile,bicone_vec,dust_vec,save_fig,orientation):

    azim, elev = orientation

    xb,yb,zb = bicone_coords
    xd,yd,zd = dust_coords
    fd = flux_profile
    vd = np.copy(vel_profile)
    vd[vd==0] = np.nan

    points = (xd,yd)
    values = zd
    # Put on uniform grid
    xdgrid,ydgrid = np.meshgrid(np.linspace(np.min(xd),np.max(xd),1000),np.linspace(np.min(yd),np.max(yd),1000))
    zdgrid = griddata(points, values, (xdgrid, ydgrid), method='cubic')

    # Truncate flux colormap
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap
    cmap = plt.get_cmap('nipy_spectral')
    new_cmap = truncate_colormap(cmap, 0.0, 0.9)

    fig = plt.figure(figsize=(14,5))
    ax4 = fig.add_subplot(1,2,1,projection='3d')
    ax5 = fig.add_subplot(1,2,2,projection='3d')
    # Plotting function for animation
    fontsize = 12
    axis_size = 2
    alpha = 1.0
    
    #################################################################################
    # Axis 4: 3D Flux Plot
    # Bicone
    flux = ax4.scatter(xb,yb,zb, marker='.',s=1,c=np.log10(fd),  alpha=0.25,zorder=10,cmap=new_cmap)
    B_arrow = Arrow3D(bicone_vec[0], bicone_vec[1], bicone_vec[2], arrowstyle="-|>", lw=3,mutation_scale=20,color='xkcd:cerulean blue',alpha=0.5)
    ax4.add_artist(B_arrow)
    # Dust Plane
    ax4.plot_wireframe(xdgrid,ydgrid,zdgrid,alpha=0.25,color='xkcd:orange',zorder=3)
    D_arrow = Arrow3D(dust_vec[0], dust_vec[1], dust_vec[2], arrowstyle="-|>", lw=3,mutation_scale=20,color='xkcd:dark orange',alpha=0.5)
    ax4.add_artist(D_arrow)
    # Axes labels annd lines
    LOS_arrow = Arrow3D([0,0], [2,1.], [0,0], arrowstyle="-|>", lw=3,mutation_scale=20,color='xkcd:red',alpha=alpha)
    ax4.add_artist(LOS_arrow)
    ax4.text(0, 2, +0.1, "L.O.S.", color='black',size=fontsize,alpha=alpha)
    # x-axis
    ax4.set_xlim(-2,2)
    ax4.set_xlabel(r'$x$',fontsize=fontsize)
    xAxisLine_pos = Arrow3D([0,axis_size], [0,0], [0,0], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    xAxisLine_neg = Arrow3D([0,-axis_size], [0,0], [0,0], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    ax4.add_artist(xAxisLine_pos)
    ax4.add_artist(xAxisLine_neg)
    ax4.text(axis_size+0.1, 0, -0.1, r'$x$', color='black',size=fontsize,alpha=alpha)
    # y-axis
    ax4.set_ylim(-2,2)
    ax4.set_ylabel(r'$y$',fontsize=fontsize)
    yAxisLine_pos = Arrow3D([0,0], [0,axis_size], [0,0], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    yAxisLine_neg = Arrow3D([0,0], [0,-axis_size], [0,0], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    ax4.add_artist(yAxisLine_pos)
    ax4.add_artist(yAxisLine_neg)
    ax4.text(0, axis_size+0.1, -0.1, r'$y$', color='black',size=fontsize,alpha=alpha)
    # z-axis
    ax4.set_zlim(-2,2)
    ax4.set_zlabel(r'$z$',fontsize=fontsize)
    zAxisLine_pos = Arrow3D([0,0], [0,0], [0,axis_size], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    zAxisLine_neg = Arrow3D([0,0], [0,0], [0,-axis_size], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    ax4.add_artist(zAxisLine_pos)
    ax4.add_artist(zAxisLine_neg)
    ax4.text(0, 0, axis_size+0.1, r'$z$', color='black',size=fontsize,alpha=alpha)
    # Add a color bar which maps values to colors.
    fig.colorbar(flux, ax=ax4, shrink=0.5, aspect=5,label=r'$\log_{10}$ Flux')
    ax4.view_init(azim=azim,elev=elev)

    #################################################################################
    # Axis 5: 3D Velocity Plot
    # Bicone
    vel = ax5.scatter(xb,yb,zb, marker='.',s=1,c=vd, cmap=cm.RdBu_r, alpha=0.25,zorder=10)
    B_arrow = Arrow3D(bicone_vec[0], bicone_vec[1], bicone_vec[2], arrowstyle="-|>", lw=3,mutation_scale=20,color='xkcd:cerulean blue',alpha=0.5)
    ax5.add_artist(B_arrow)
    # Dust Plane
    ax5.plot_wireframe(xdgrid,ydgrid,zdgrid,alpha=0.25,color='xkcd:orange',zorder=3)
    D_arrow = Arrow3D(dust_vec[0], dust_vec[1], dust_vec[2], arrowstyle="-|>", lw=3,mutation_scale=20,color='xkcd:dark orange',alpha=0.5)
    ax5.add_artist(D_arrow)
    # Axes labels annd lines
    LOS_arrow = Arrow3D([0,0], [2,1.], [0,0], arrowstyle="-|>", lw=3,mutation_scale=20,color='xkcd:red',alpha=alpha)
    ax5.add_artist(LOS_arrow)
    ax5.text(0, 2, +0.1, "L.O.S.", color='black',size=fontsize,alpha=alpha)
    # x-axis
    ax5.set_xlim(-2,2)
    ax5.set_xlabel(r'$x$',fontsize=fontsize)
    xAxisLine_pos = Arrow3D([0,axis_size], [0,0], [0,0], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    xAxisLine_neg = Arrow3D([0,-axis_size], [0,0], [0,0], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    ax5.add_artist(xAxisLine_pos)
    ax5.add_artist(xAxisLine_neg)
    ax5.text(axis_size+0.1, 0, -0.1, r'$x$', color='black',size=fontsize,alpha=alpha)
    # y-axis
    ax5.set_ylim(-2,2)
    ax5.set_ylabel(r'$y$',fontsize=fontsize)
    yAxisLine_pos = Arrow3D([0,0], [0,axis_size], [0,0], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    yAxisLine_neg = Arrow3D([0,0], [0,-axis_size], [0,0], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    ax5.add_artist(yAxisLine_pos)
    ax5.add_artist(yAxisLine_neg)
    ax5.text(0, axis_size+0.1, -0.1, r'$y$', color='black',size=fontsize,alpha=alpha)
    # z-axis
    ax5.set_zlim(-2,2)
    ax5.set_zlabel(r'$z$',fontsize=fontsize)
    zAxisLine_pos = Arrow3D([0,0], [0,0], [0,axis_size], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    zAxisLine_neg = Arrow3D([0,0], [0,0], [0,-axis_size], arrowstyle="-|>", lw=1,mutation_scale=10,color='xkcd:black',alpha=alpha)
    ax5.add_artist(zAxisLine_pos)
    ax5.add_artist(zAxisLine_neg)
    ax5.text(0, 0, axis_size+0.1, r'$z$', color='black',size=fontsize,alpha=alpha)
    # Add a color bar which maps values to colors.
    fig.colorbar(vel, ax=ax5, shrink=0.5, aspect=5,label=r'Projected Velocity along LOS (km/s)')
    ax5.view_init(azim=azim,elev=elev)

    plt.tight_layout()

    if save_fig==True:
        # plt.savefig('model_3d.pdf',dpi=150,fmt='pdf')
        plt.savefig('model_3d.png',dpi=300,fmt='png')
    # plt.close()
    return None


#####################################################################################

def emission_model(fgrid,vgrid,vmax,obs_res=68.9,nbins=25,sampling=100,plot=True,save_fig=True):  
    
    if int(sampling)%2==0:
            # Having an odd sampling ensures that there is a value at (0,0,0)
            sampling = int(sampling)+1

    # Reshape grids into cubes
    fgrid = fgrid.reshape(sampling,sampling,sampling)
    vgrid = vgrid.reshape(sampling,sampling,sampling)
            
    def gaussian(x, f, vel, sig):
        return f * 1.0/(sig * np.sqrt(2.0*np.pi)) * np.exp(-np.power(x - vel, 2.0) / (2.0 * np.power(sig, 2.0)) ) # 
        
    def get_bin_centers(bins):
            bins = bins[:-1]
            bin_width = bins[1]-bins[0]
            new_bins =  bins + bin_width/2.0
            return new_bins

    
    losvd = [] # an array to hold the losvd at every non-zero pixel
    bins = np.linspace(-vmax,vmax,nbins)
    # t0 = time.time()
    for i in range(sampling):
        for j in range(sampling):
            v_xy = vgrid[:,i,j]
            f_xy = fgrid[:,i,j]
            v_xy = v_xy[f_xy>0]
            f_xy = f_xy[f_xy>0]
            if (len(v_xy)>0) and (len(f_xy)>0):
                v_hist,v_edges = np.histogram(v_xy,bins=bins,weights=f_xy)
                losvd.append(v_hist)
                
    # print('\n      %0.9f seconds' % float(time.time()-t0))
    
    losvd = (np.sum(losvd,axis=0)) # sum over x and y
    losvd = losvd/np.max(losvd) # normalize to 1
    # Interpolate the losvd histogram so it is smooth
    x = np.arange(-vmax,vmax+1,1)
    losvd_interp = interp1d(bins[:-1],losvd,kind='cubic',bounds_error=False,fill_value='extrapolate')
    emline = losvd_interp(x)
    # Convolve with gaussian centered at zero 
    emline = gaussian_filter(emline,sigma=obs_res)
    # normalize to 1
    emline = emline/np.max(emline) 
    # convert x in km/s to angstroms
    c = 299792. # speed of light in km/s
    cw = 5008.240# central wavelength; [OIII]5007 (SDSS)
    x_ang = cw+(x*cw)/c

    # Plot
    if plot==True:
        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        # angstroms
        ax1.plot(x_ang,emline,color='red',label=r'$\sigma_{\rm{SDSS}}=$%0.1f (km s$^{-1}$)' % obs_res)
        ax1.bar(cw+(v_edges[:-1]*cw/c), losvd, ec="k", align='center', width=np.diff(cw+(v_edges[:-1]*cw/c))[0])
        ax1.axvline(cw,color='black',linestyle='--',linewidth=0.5)
        ax1.axhline(0.0,color='black',linestyle='--',linewidth=0.5)
        ax1.axhline(1.0,color='black',linestyle='--',linewidth=0.5)
        ax1.set_ylim(0,1.1)
        ax1.set_xlabel(r'$\lambda_{\rm{rest}}$ ($\rm{\AA}$)')
        ax1.set_ylabel(r'Normalized Flux')
        ax1.legend(loc='best')
        # km/s 
        ax2.plot(x,emline,color='red',label=r'$\sigma_{\rm{obs}}=$%0.1f (km s$^{-1}$)' % obs_res)
        ax2.bar(v_edges[:-1], losvd, width=np.diff(v_edges), ec="k", align="center")
        ax2.axvline(0.0,color='black',linestyle='--',linewidth=0.5)
        ax2.axhline(0.0,color='black',linestyle='--',linewidth=0.5)
        ax2.axhline(1.0,color='black',linestyle='--',linewidth=0.5)
        ax2.set_ylim(0,1.1)
        ax2.set_xlabel(r'Velocity (km s$^{-1}$)')
        ax2.set_ylabel(r'Normalized Flux')
        ax2.legend(loc='best')
        plt.tight_layout()
        if save_fig==True:
            # plt.savefig('emission_model.pdf',dpi=150,fmt='pdf')
            plt.savefig('emission_model.png',dpi=300,fmt='png')
        # plt.close()
    return x_ang, emline

#### Convert Seconds to Minutes #####################################################

# Python Program to Convert seconds 
# into hours, minutes and seconds 
  
def time_convert(seconds): 
    seconds = seconds % (24. * 3600.) 
    hour = seconds // 3600.
    seconds %= 3600.
    minutes = seconds // 60.
    seconds %= 60.
      
    return "%d:%02d:%02d" % (hour, minutes, seconds)

##################################################################################

### Least Squares function #############################################################

def likelihood(params,param_names, x, y, yerr):
    pdict = {}
    for k in range(0,len(param_names),1):
        pdict[param_names[k]] = params[k]
    A             = pdict['A']
    tau           = pdict['tau']
    theta_in_deg  = pdict['theta_in_deg']
    theta_out_deg = pdict['theta_out_deg'] 
    theta_B1_deg  = pdict['theta_B1_deg'] 
    theta_D1_deg  = pdict['theta_D1_deg'] 
    vmax          = pdict['vmax']

    model = bicone_model(x, A, tau, theta_in_deg, theta_out_deg,
                         theta_B1_deg,
                         theta_D1_deg,
                         vmax)
    
#     return np.sum((y-model)**2)
    return np.sum(-np.log(yerr*np.sqrt(2*np.pi))-0.5*(y-model)**2/yerr**2)


### Make model #######################################################################

def bicone_model(wave, A, tau, theta_in_deg, theta_out_deg,
                 theta_B1_deg,
                 theta_D1_deg,
                 vmax):
    # Constant parameters
    D             = 1.0  # length of bicone (arbitrary units)
    fn            = 1.0e3 # initial flux value at center
    theta_B2_deg  = 0.0 # Lock-out LOS y-axis rotation (symmetry)
    theta_B3_deg  = 0.0 # Lock-out the z-axis rotation (symmetry)
#     theta_D1_deg  = 90.0 # Lock-out the x-axis rotation (type 1 AGN)
    theta_D2_deg  = 0.0 # Lock-out the y-axis rotation (symmetry)
    theta_D3_deg  = 0.0 # Lock-out the z-axis rotation (symmetry)
    vtype='decreasing' # 'increasing','decreasing', or 'constant'
    # Sampling paramters
    sampling = 50 # minimum point sampling
    map_interpolation = 'none'
    obs_res = 68.9 # resolution of SDSS for emission line model
    nbins= 40
    # Bicone coordinate, flux, and velocity grids
    xbgrid,ybgrid,zbgrid,fgrid,vgrid = generate_bicone(theta_in_deg, theta_out_deg,
                                                       theta_B1_deg, theta_B2_deg, theta_B3_deg,
                                                       theta_D1_deg, theta_D2_deg, theta_D3_deg,
                                                       D=D, tau=tau, fn=fn, A=A,
                                                       vmax=vmax, vtype=vtype,
                                                       sampling=sampling,plot=False,save_fig=False)
    # Get emission line model
    x,emline = emission_model(fgrid,vgrid,vmax=vmax,obs_res=obs_res,nbins=nbins,sampling=sampling,
                              plot=False,save_fig=False)
    # Interpolate emission line model onto wavelength grid
    interp = interp1d(x,emline,bounds_error=False,fill_value=0.001)
    model = interp(wave)
            
    return model

##################################################################################

