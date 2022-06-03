import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def scatter_plot_kde(df, columnX, columnY, output_name=None,output_directory=None, force_n_to_cero = True, x_label = None, y_label = None, 
                     same_axes=False, linear_fit=False, x_lims=None, y_lims=None, x_ticks=None, y_ticks=None,figsize=(15,5),
                     highlights1=None,highlights2=None, hl1_color='r', hl2_color='c', highlights_kdw={'s': 50, 'edgecolor': 'k', 'alpha':1},
                     colorbar=False, watermark=None,True_KDE=False, prdw_logo_path=None,
                     prdw_logo_size=[0.79, 0.80, 0.1, 0.1], prdw_logo_alpha=0.5,large_ticks=True, x_tick_labels = None, y_tick_labels = None):
    '''Plots and saves figure of a Kernel Density Estimate (kde) scatter. A maximum of 3 figures can be plotted as subplots. 
        Locations of subplots are in the following order: NW,NE,SE. 
    
    Parameters:
        df: dataframe
            Dataframe with information. 
        columnX: string or list of strings
            Name of column(s) to be plotted in X axis 
        columnY: string or list of strings
            Name of column(s) to be plotted in Y axis
        force_n_to_cero: boolean (optional)
            True if Y crossing at (0,0) is imposed, False otherwise. Default= True
        x_label: string or list of strings (optional)
            X axis label(s)
        y_label: string or list of strings(optional)
            Y axis label(s)
        same_axes: boolean (optional)
            True if X & Y axis have similar scales. e.g., when plotting Hm0 vs Direction, False is recommended. Default= False
        linear_fit: boolean (optional)
            True if a linear fit (regression) is wanted. Default = False
        x_lims: list or list of list (optional)
            list with X axis limits, [xmin, xmax]
        y_lims: list or list of list (optional)
            list with Y axis limits, [ymin, ymax]
        x_ticks: list or list of list (optional)
            list with X axis ticks, eg. np.arange(xmin, xmax, dx)
        y_ticks: list or list of list (optional)
            list with Y axis ticks, eg. np.arange(ymin, ymax, dy)       
        output_name: string (optional)
            Output file name
        output_directory: string (optional)
            Output file path
        figsize: tuple (optional)
            Figure size (width,height). Default= (15,5) 
        highlights1: list (optional)
            List of tuples in the form [(x0,y0),(x1,y1),...,(xn,yn)] to be plotted on top of scatter. 
            If working with more than one subplot, highlights should be a list of list of tuples
        highlights2: list (optional)
            idem highlights1
        hl1_color: string
            color of the highlights1 points
        hl2_color: string
            color of the highlights2 points
        highlights_size: int
            size of the highlighted points
        colorbar: boolean (optional)
            Displays colorbars. Default: False
        watermark: str (optional)
            Displays watermark. Default: None
        True_KDE: Boolean (optional)
            Displays the True KDE (True, this is very slow) or a 2D Histogram proxy (False, this is fast). Default: False
        prdw_logo_path: str (optional). Defualt: None
            Complete path of the PRDW logo. It is stored on the gitlab documentation folder
        prdw_logo_size: list (optional). Default: [0.13, 0.61, 0.08, 0.08]
            [x0, y0, width, height]. The aspect of the logo is set to equal.
        prdw_logo_alpha: float (optional).
            Default: 0.5
        large_ticks: string (optional)
            Ticks font size Extra large. Default=True.
    ------------------------------------------------------------------------------------------------
    Returns:
        ax: Matplotlib axes
    '''
    if type(columnX)==str: #Code for 1 subplot       
        x = df.loc[:, columnX].values
        y = df.loc[:, columnY].values
        
        if True_KDE==True:
            xy = np.vstack([x,y])
            z = gaussian_kde(xy)(xy)
        else:
            z = scatter_interpolate(x,y)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        if force_n_to_cero == True:
            x2 = x[:,np.newaxis]
            m, _, _, _ = np.linalg.lstsq(x2, y)
            y2 = m*x2
            r2 = max(r2_score(y, y2), 0)
            text1 = 'm = '+str(np.round(m[0],2))
            text2 = 'r2 ='+str(np.round(r2,2))
            text3 = 'n = 0'
        else:
            x2 = x
            A = np.vstack([x2, np.ones(len(x2))]).T
            m, c = np.linalg.lstsq(A, y)[0]
            y2 = m*x2+c
            r2 = max(r2_score(y, y2), 0)
            text1 = 'm = '+str(np.round(m,2))
            text2 = 'r2 ='+str(np.round(r2,2))
            text3 = 'n = '+str(np.round(c,2))

        minvalue = min(int(x.min()), int(y.min()))
        maxvalue = max(int(x.max())+1, int(y.max()+1))

        #extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        fig, ax =  plt.subplots(figsize = figsize)
        cm = plt.cm.get_cmap('viridis')
        sc = ax.scatter(x, y, c=z, edgecolor='', cmap = cm, alpha = 0.5)
        if linear_fit == True:
            ax.legend([extra, extra, extra],[text1, text2, text3], loc='upper left')
            ax.plot(x2, y2, 'w-')
        else:
            pass

        if (x_label is None) and (y_label is None):
            ax.set_xlabel(columnX, fontsize = 'x-large')
            ax.set_ylabel(columnY, fontsize = 'x-large')
        else:
            ax.set_xlabel(x_label, fontsize = 'x-large')
            ax.set_ylabel(y_label, fontsize = 'x-large')
        if same_axes == True:
            ax.set_ylim([minvalue, maxvalue])
            ax.set_xlim([minvalue, maxvalue])
        else:
            pass

        ax.grid(True)
        
        if x_lims is not None:
            ax.set_xlim(x_lims)
        if y_lims is not None:
            ax.set_ylim(y_lims)
        if x_ticks is not None:
            ax.set_xticks(x_ticks);
        if y_ticks is not None:
            ax.set_yticks(y_ticks);
        if x_tick_labels is not None:
            ax.set_xticklabels(x_tick_labels);
        if y_tick_labels is not None:
            ax.set_yticklabels(y_tick_labels);
        
        if large_ticks==True:
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize('x-large')
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize('x-large')
            
        if watermark != None:
            text = AnchoredText(watermark, 'upper right',frameon = False, borderpad = -2, prop=dict(fontsize = 'xx-small', alpha = 0.4)) ;
            ax.add_artist(text)
        
        if highlights1!=None: #Plot highlights 
            x_high1,y_high1=list(zip(*highlights1))
            ax.scatter(x_high1, y_high1, c=hl1_color, **highlights_kdw)    
            
        if highlights2!=None: #Plot highlights 
            x_high2,y_high2=list(zip(*highlights2))
            ax.scatter(x_high2, y_high2, c=hl2_color, **highlights_kdw)
            
        if colorbar==True:
            normalize = mpl.colors.Normalize(vmin=min(z), vmax=max(z))
            cax, _ = mpl.colorbar.make_axes(ax)
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=cm,norm=normalize)
            cbar.set_label('Probability density', labelpad = -60, y=0.55, rotation=90)
        
        if prdw_logo_path != None:
            im = plt.imread(prdw_logo_path)
            ax2 = fig.add_axes(prdw_logo_size)
            ax2.imshow(im, aspect = 'equal', alpha = prdw_logo_alpha)
            ax2.axis('off');
        
    elif len(columnX)==2: #Code for 2 subplots
        fig,ax =  plt.subplots(1,2,figsize = figsize)
        for sbplt in range(len(columnX)):
            x = df.loc[:, columnX[sbplt]].values
            y = df.loc[:, columnY[sbplt]].values            
            if True_KDE==True:
                xy = np.vstack([x,y])
                z = gaussian_kde(xy)(xy)
            else:
                z = scatter_interpolate(x,y)            

            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]

            if force_n_to_cero == True:
                x2 = x[:,np.newaxis]
                m, _, _, _ = np.linalg.lstsq(x2, y)
                y2 = m*x2
                r2 = max(r2_score(y, y2), 0)
                text1 = 'm = '+str(np.round(m[0],2))
                text2 = 'r2 ='+str(np.round(r2,2))
                text3 = 'n = 0'
            else:
                x2 = x
                A = np.vstack([x2, np.ones(len(x2))]).T
                m, c = np.linalg.lstsq(A, y)[0]
                y2 = m*x2+c
                r2 = max(r2_score(y, y2), 0)
                text1 = 'm = '+str(np.round(m,2))
                text2 = 'r2 ='+str(np.round(r2,2))
                text3 = 'n = '+str(np.round(c,2))

            minvalue = min(int(x.min()), int(y.min()))
            maxvalue = max(int(x.max())+1, int(y.max()+1))

            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

            cm = plt.cm.get_cmap('viridis')
            sc = ax[sbplt].scatter(x, y, c=z, edgecolor='', cmap = cm, alpha = 0.5)
            
            ##Colorbar
            if colorbar==True:
                normalize = mpl.colors.Normalize(vmin=min(z), vmax=max(z))
                cax, _ = mpl.colorbar.make_axes(ax[sbplt])
                cbar = mpl.colorbar.ColorbarBase(cax, cmap=cm,norm=normalize)
                cbar.set_label('Probability density', labelpad=-50, y=0.55, rotation=90)
            
            if linear_fit == True:
                ax[sbplt].legend([extra, extra, extra],[text1, text2, text3], loc='upper left')
                ax[sbplt].plot(x2, y2, 'w-')
            else:
                pass

            if (x_label is None) and (y_label is None):
                ax[sbplt].set_xlabel(columnX[sbplt], fontsize = 'x-large')
                ax[sbplt].set_ylabel(columnY[sbplt], fontsize = 'x-large')
            else:
                ax[sbplt].set_xlabel(x_label[sbplt], fontsize = 'x-large')
                ax[sbplt].set_ylabel(y_label[sbplt], fontsize = 'x-large')
            if same_axes == True:
                ax[sbplt].set_ylim([minvalue, maxvalue])
                ax[sbplt].set_xlim([minvalue, maxvalue])
            else:
                pass
            if x_lims is not None:
                ax[sbplt].set_xlim(x_lims[sbplt])
            if y_lims is not None:
                ax[sbplt].set_ylim(y_lims[sbplt])
                
            if x_ticks is not None:
                ax[sbplt].set_xticks(x_ticks[sbplt]);
            if y_ticks is not None:
                ax[sbplt].set_yticks(y_ticks[sbplt]);
            if x_tick_labels is not None:
                ax[sbplt].set_xticklabels(x_tick_labels[sbplt]);
            if y_tick_labels is not None:
                ax[sbplt].set_yticklabels(y_tick_labels[sbplt]);
                
                
            if large_ticks==True:
                for tick in ax[sbplt].xaxis.get_major_ticks():
                    tick.label.set_fontsize('x-large')
                for tick in ax[sbplt].yaxis.get_major_ticks():
                    tick.label.set_fontsize('x-large')

            if highlights1!=None: #Plot highlights 
                x_high1,y_high1=list(zip(*highlights1[sbplt]))
                ax.scatter(x_high1, y_high1, c=hl1_color, **highlights_kdw)
            if highlights2!=None: #Plot highlights 
                x_high2,y_high2=list(zip(*highlights2[sbplt]))
                ax.scatter(x_high2, y_high2, c=hl1_color, **highlights_kdw)

        if watermark != None:
            text = AnchoredText(watermark, 'upper right',frameon = False, borderpad = -2, prop=dict(fontsize = 'xx-small', alpha = 0.4)) ;
            ax[1].add_artist(text)
            
        if prdw_logo_path != None:
            im = plt.imread(prdw_logo_path)
            ax2 = fig.add_axes(prdw_logo_size)
            ax2.imshow(im, aspect = 'equal', alpha = prdw_logo_alpha)
            ax2.axis('off');
            
    else: #Code for 3 subplots
        gs = gridspec.GridSpec(2, 4)
        gs.update(wspace=0.5)
        fig = plt.figure(figsize=(15,10))
        for sbplt in range(len(columnX)):
            if sbplt==0:
                ax=plt.subplot(gs[0, :2])
            elif sbplt==1:
                ax=plt.subplot(gs[0, 2:])
                if watermark != None:
                    text = AnchoredText(watermark, 'upper right',frameon = False, borderpad = -2, prop=dict(fontsize = 'xx-small', alpha = 0.4)) ;
                    ax.add_artist(text)
            else:
                ax=plt.subplot(gs[1, 2:])
            x = df.loc[:, columnX[sbplt]].values
            y = df.loc[:, columnY[sbplt]].values
            
            
            if True_KDE==True:
                xy = np.vstack([x,y])
                z = gaussian_kde(xy)(xy)
            else:
                z = scatter_interpolate(x,y)

            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]

            if force_n_to_cero == True:
                x2 = x[:,np.newaxis]
                m, _, _, _ = np.linalg.lstsq(x2, y)
                y2 = m*x2
                r2 = max(r2_score(y, y2), 0)
                text1 = 'm = '+str(np.round(m[0],2))
                text2 = 'r2 ='+str(np.round(r2,2))
                text3 = 'n = 0'
            else:
                x2 = x
                A = np.vstack([x2, np.ones(len(x2))]).T
                m, c = np.linalg.lstsq(A, y)[0]
                y2 = m*x2+c
                r2 = max(r2_score(y, y2), 0)
                text1 = 'm = '+str(np.round(m,2))
                text2 = 'r2 ='+str(np.round(r2,2))
                text3 = 'n = '+str(np.round(c,2))

            minvalue = min(int(x.min()), int(y.min()))
            maxvalue = max(int(x.max())+1, int(y.max()+1))

            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

            cm = plt.cm.get_cmap('viridis')
            sc = ax.scatter(x, y, c=z, edgecolor='', cmap = cm, alpha = 0.5)
            ##Colorbar
            if colorbar==True:
                normalize = mpl.colors.Normalize(vmin=min(z), vmax=max(z))
                cax, _ = mpl.colorbar.make_axes(ax)
                cbar = mpl.colorbar.ColorbarBase(cax, cmap=cm,norm=normalize)
                cbar.set_label('Probability density', labelpad=-50, y=0.55, rotation=90)
            
            if linear_fit == True:
                ax.legend([extra, extra, extra],[text1, text2, text3], loc='upper left')
                ax.plot(x2, y2, 'w-')
            else:
                pass

            if (x_label is None) and (y_label is None):
                ax.set_xlabel(columnX[sbplt], fontsize = 'x-large')
                ax.set_ylabel(columnY[sbplt], fontsize = 'x-large')
            else:
                ax.set_xlabel(x_label[sbplt], fontsize = 'x-large')
                ax.set_ylabel(y_label[sbplt], fontsize = 'x-large')
            if same_axes == True:
                ax.set_ylim([minvalue, maxvalue])
                ax.set_xlim([minvalue, maxvalue])
            else:
                pass
                
            if x_lims is not None:
                ax.set_xlim(x_lims[sbplt])
            if y_lims is not None:
                ax.set_ylim(y_lims[sbplt])
                
            if x_ticks is not None:
                ax.set_xticks(x_ticks[sbplt])
            if y_ticks is not None:
                ax.set_yticks(y_ticks[sbplt])  
            if x_tick_labels is not None:
                ax.set_xticklabels(x_tick_labels[sbplt]);
            if y_tick_labels is not None:
                ax.set_yticklabels(y_tick_labels[sbplt]);
            
            if large_ticks==True:
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize('x-large')
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize('x-large')
                
            if highlights1!=None: #Plot highlights 
                x_high1,y_high1=list(zip(*highlights1[sbplt]))
                ax.scatter(x_high1, y_high1, c=hl1_color, **highlights_kdw)
            if highlights2!=None: #Plot highlights 
                x_high2,y_high2=list(zip(*highlights2[sbplt]))
                ax.scatter(x_high2, y_high2, c=hl2_color, **highlights_kdw)
            #Add minor grid lines
            ax.minorticks_on()
            ax.grid(which='minor', linestyle='--', linewidth='0.5', color='w')
        if prdw_logo_path != None:
            im = plt.imread(prdw_logo_path)
            ax2 = fig.add_axes(prdw_logo_size)
            ax2.imshow(im, aspect = 'equal', alpha = prdw_logo_alpha)
            ax2.axis('off');

            
    ##Save output
    if output_name!=None:
        if output_directory==None:
            plt.savefig(output_name, dpi = 500, bbox_inches = 'tight')
        else:
            plt.savefig(os.path.join(output_directory, output_name), dpi = 500, bbox_inches = 'tight')
            
    return ax

def scatter_interpolate(x,y,res=17,method='cubic',normalise=True, return_res = False):
    '''Interpolates scatter data on a 2D histogram based on density. This is a proxy of the KDE estimator, with lower (still acceptable) resolution but much faster.
    
    Parameters:
        x: Pandas series or array
            Series with X data
        y: Pandas series or array
            Series with Y data
        res: Int (Default=17)
            Grid resolution.
        method: Str (Default='cubic')
            Scipy griddata interpolation method.
        normalise: Boolean (Default= False)
            True: Normalize data to probability density Probability/unit_X*unit_y
            False: Point density per unit area Points/unit_X*unit_y
        return_res: bool (default: False)
            If True: returns gridZ, dx, dy
            If False: returns gridZ
    ------------------------------------------------------------------------------------------------
    Returns:
        gridZ: Array
            Array with the colors based on histogram density
    '''
    

    # Setup a lin-grid for interpolation
    ## Centre points
    cx = np.linspace(min(x), max(x), res)
    cy = np.linspace(min(y), max(y), res)
    
    ## Resolution
    dx = cx[1]-cx[0]
    dy = cy[1]-cy[0]
    
    ## Edege points
    ex = np.linspace(min(x)-dx*0.5, max(x)+dx*0.5, res+1)
    ey = np.linspace(min(y)-dy*0.5, max(y)+dy*0.5, res+1)
        
    # Numpy method for calculating 2D histogram
    histoData, exh, eyh = np.histogram2d(x,y,[ex,ey],normed=normalise)

    # EXTRACT HISTOGRAM VALIUES
    # unpacking data
    h = []
    for j in range(len(cy)):
        for i in range(len(cx)):
            h.append(histoData[i,j])        
    
    # PRE-PROCESS GRID DATA
    xg, yg = np.meshgrid(cx, cy)
    xg = xg.ravel()
    yg = yg.ravel()
    
    ## Interpolate histogram density data to original scatter
    gridZ = interpolate.griddata((xg, yg),h, (x,y),method=method,rescale=True)
        
    if return_res:
        return gridZ, dx, dy
    else:
        return gridZ
