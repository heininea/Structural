# Strain and stress field of a Hertzian plane strain contact
# based on K.L. Johnson's "Contact mechanics" (1987, Cambridge university press)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
    
def main():
  
    #Parameters
    dg = 1.26       # Grain diamater
    a = dg / 2      # Half of the grain diameter
    Yield = 1240       # maximum Hertzian pressure (MPa)           #Material Yield Limit in our case
    C = 1.2
    p0 = C*1240
    mu = 0.24        # Coefficient of grinding (Ft/Fn)
    nu = 0.3        # Workpiece Poisson ratio
    E = 2.15*10**5  # Workpiece Young's modulus
    
    #Parameters of the simulation
    nbPts_s = 200
    nbPts_x = 200
    nbPts_z = 200
    
    #Rectangle limits
    xl = -5
    xh = +5
    zl = 0.0
    zh = 2.5
    
    #Define x, z and s tridimensional arrays with meshgrid function
    x_ = np.linspace(xl, xh, num=nbPts_x)
    z_ = np.linspace(zl, zh, num=nbPts_z)
    s_ = np.linspace(-a, +a, num=nbPts_s)
    x, z, s = np.meshgrid(x_, z_, s_)
    
    #Compute components of the stresses
    ps = p0*np.sqrt(1-(s/a)**2)
    norm_x     = ps *(x-s)**2/((x-s)**2+z**2)**2
    norm_z     = ps/((x-s)**2+z**2)**2
    norm_shear = ps*(x-s)/((x-s)**2+z**2)**2
    tang_x     = mu*ps*(x-s)**3/((x-s)**2+z**2)**2
    tang_z     = mu*ps*(x-s)/((x-s)**2+z**2)**2
    tang_shear = mu*ps*(x-s)**2/((x-s)**2+z**2)**2   
    
    #Compute stresses on the whole mesh    
    sigma_xx = (-2 / np.pi) * (z_[:,None] * np.trapz(norm_x, x=s, axis=2) + np.trapz(tang_x, x=s, axis=2))
    sigma_zz = (-2 / np.pi) * (z_[:,None]**3 * np.trapz(norm_z, x=s, axis=2) + z_[:,None]**2 * np.trapz(tang_z, x=s, axis=2))
    sigma_xz = (-2 / np.pi) * (z_[:,None]**2 * np.trapz(norm_shear, x=s, axis=2) + z_[:,None] * np.trapz(tang_shear, x=s, axis=2))
    
    #Compute the stress on the surface of the workpiece when Z=0
    sigma_zz[0, (x_ > -a) & (x_ < a)] = -p0 * np.sqrt(1 - (x_[(x_ > -a) & (x_ < a)] / a)**2)
    sigma_xz[0, (x_ > -a) & (x_ < a)] = -mu * p0 * np.sqrt(1 - (x_[(x_ > -a) & (x_ < a)] / a)**2)
    sigma_xx[0, (x_ > -a) & (x_ < a)] = -p0 * np.sqrt(1 - (x_[(x_ > -a) & (x_ < a)] / a)**2)
    sigma_zz[0, np.logical_not((x_ > -a) & (x_ < a))] = 0
    sigma_xz[0, np.logical_not((x_ > -a) & (x_ < a))] = 0
    sigma_xx[0, np.logical_not((x_ > -a) & (x_ < a))] = 0 
    
    #Compute sigmayy using the Poisson's ratio of the workpiece
    sigma_yy = nu * (sigma_xx + sigma_zz)
    
    # Calculate the equivalent von Mises stress  
    sigma_vm = np.sqrt(0.5*(sigma_xx - sigma_yy)**2 + (sigma_yy - sigma_zz)**2 + (sigma_zz - sigma_xx)**2 + 6*sigma_xz**2)
    
    # Compute strains  
    eps_xx = 1/E*((1-nu**2)*sigma_xx -nu*(1+nu)*sigma_zz)
    eps_zz = 1/E*((1-nu**2)*sigma_zz -nu*(1+nu)*sigma_xx)
    gamma_xz = 2*(1+nu)/E*sigma_xz
    
    #
    print("Max sigmaxx  ", np.max(sigma_xx))
    print("Min sigmaxx: ", np.min(sigma_xx))
    print("Max sigmazz: ", np.max(sigma_zz))
    print("Min sigmazz: ", np.min(sigma_zz))
    print("Max sigmaxz: ", np.max(sigma_xz))
    print("Min sigmaxz: ", np.min(sigma_xz))
    print("Max Von Mises: ", np.max(sigma_vm))
    print("Max Von Mises vs. Yield strength: ", np.max(sigma_vm)/Yield)
    #Plot sigma_xx
    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    ax.pcolormesh(x[:,:,0], z[:,:,0], sigma_xx, cmap='jet', antialiased=True)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('Sigma_xx (MPa)')
 
    #Plot sigma_yy
    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    ax.pcolormesh(x[:,:,0], z[:,:,0], sigma_yy, cmap='jet', antialiased=True)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('Sigma_yy (MPa)')
    
    #Plot sigma_zz
    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    ax.pcolormesh(x[:,:,0], z[:,:,0], sigma_zz, cmap='jet', antialiased=True)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('Sigma_zz (MPa)')
    
    #Plot sigma_xz
    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    ax.pcolormesh(x[:,:,0], z[:,:,0], sigma_xz, cmap='jet', antialiased=True)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('Sigma_xz (MPa)')
    
    #Plot sigma_xz
    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    ax.pcolormesh(x[:,:,0], z[:,:,0], eps_xx, cmap='jet', antialiased=True)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('Eps_xx (-)')

    #Plot sigma_xz
    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    ax.pcolormesh(x[:,:,0], z[:,:,0], eps_zz, cmap='jet', antialiased=True)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('Sigma_xz (MPa)')
    
    #Plot sigma_xz
    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    ax.pcolormesh(x[:,:,0], z[:,:,0], gamma_xz, cmap='jet', antialiased=True)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('Gamma_xz (MPa)')
    

main()           