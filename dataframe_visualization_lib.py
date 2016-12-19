import seaborn as sns
sns.set(style="white",color_codes=True)
import inspect
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pandas as pd

def infer_feature_types(df,suppress_report=False):
    """
    Returns dictionary of features and corresponding datatypes 
    (Either 'category' or 'numerical') given a dataframe
    
    This can easily misclassify integers, since they can represent 
    either counts or categories. Additionally, one should try to 
    avoid using floats as integers.
    
    A summary is printed at the end so the user can manually modify
    the resulting dictionary.        
    """
    # Initialize output dictionary
    data_types = {}

    # Consider each feature
    for feature in list(df.columns):
        # Obtain unique feature values
        unique_feature_values = list(set(df[feature]))
        
        # Obtain unique feature types
        unique_feature_types = list(set(type(unique_item) for unique_item in unique_feature_values))

        # Check for mixed types within feature
        if len(unique_feature_types) != 1:
            raise Exception("Mixed types in feature '%s' encountered. This is presently unsupported."%(feature))

        # Check that all input types are expected
        unique_feature_type = unique_feature_types[0]
        if not (unique_feature_type is not str or unique_feature_type is not int or unique_feature_type is not np.int64 or unique_feature_type is not float or unique_feature_type is not np.float64):
            raise Exception("Feature '%s' is not of type str, int, numpy.int64, float, or numpy.float64. Its listed feature is %s This is currently unsupported"%(feature,unique_feature_type))
        
        # Ignore features that appear to be ids, though warn user
        if unique_feature_type is str or unique_feature_type is int or unique_feature_type is np.int64:
            if len(unique_feature_values) != len(df[feature]):
                data_types[feature] = 'category'
            else:
                if not suppress_report:
                    print "WARNING: feature '%s' appears to be an id and will not be included in the plot"%(feature)
        else:
            data_types[feature] = 'numerical'
    
    # Print report
    if not suppress_report:
        print ''
        print 40*'-'
        print 'Data type\tFeature'
        print 40*'-'    
        for key in data_types:
            print '%s\t%s'%(data_types[key],key)
        print 40*'-'
    
    # Return
    return data_types

def get_color_val(ind,num_series):
    colormap = 'rainbow'
    color_map = plt.get_cmap(colormap)
    
    custom_map = ['grey','orange','red','purple','blue','cyan','lime','yellow','black']
    
    # Calculate color
    if num_series > len(custom_map):
        if not ind:
            colorVal = 'gray'
        else:
            colorVal = color_map((ind-1)/float(num_series+1))
    else:
        colorVal = custom_map[ind]
    
    return colorVal

def compare_data(df,plot_vars=[],data_types=[],bar_alpha=0.85,
                 num_bars=20,palette=['grey','orange','red'],fig_size=60,
                 fig_aspect=1):
    """
    Outputs a modified pair grid where data plots takes this form:
    
    same feature vs same feature: Uncolored distribution
    continuous vs continuous: scatter plot colored by the default feature
    category vs continuous: distribution of continuous feature colored by category
    category 1 vs category 2: distribution of category 2 colored by category 1
    continuous vs category: As yet undetermined... probably blank
    """

    # Use all features if not explicitly provided by user
    if not plot_vars:
        plot_vars = list(df.columns)
        
    # Check that data_types are specified
    if not data_types:
        raise Exception('Dictionary of feature:data types keyword argument, data_types, must be specified.')
    
    # Keep only the plotting Filter plot_vars so only those included in data_types is included
    plot_vars = [plot_var for plot_var in plot_vars if plot_var in data_types]
    
    # Count number of features
    number_features = len(plot_vars)
    
    # Set text and line color
    grayLevel = 0.6
    text_and_line_color = (0.0,0.0,0.0,grayLevel)
    
    # Set default text font, color, and size
    text_family = 'sans-serif'
    text_font = 'Helvetica Neue Light'
    text_font_size = 8
    label_size = text_font_size-2
    
    # set default text font, color, and size
    matplotlib.rc('font',family=text_family) 
    matplotlib.rc('font',serif=text_font)
    matplotlib.rcParams['text.color'] = text_and_line_color
    matplotlib.rcParams['font.size'] = text_font_size
    
    # Set bar parameters
    bar_width = 0.4
    
    # Generate figure
    fig = plt.figure(figsize=[fig_size,fig_size*fig_aspect])
    
    # Populate axes
    axes = []
    for axis_row_ind in range(number_features):
        # Initialize current row of axes
        axes.append([])
        for axis_column_ind in range(number_features):
            # Create subplot
            axes[-1].append(fig.add_subplot(fig_size,fig_size*fig_aspect,axis_row_ind*fig_size*fig_aspect+axis_column_ind+1))
            
            # Get the feature names for the current row/column
            row_feature = plot_vars[axis_row_ind]
            col_feature = plot_vars[axis_column_ind]
            
            # Get the feature type (categorical or numerical) for current row/column
            row_type = data_types[row_feature]
            col_type = data_types[col_feature]
            
            # Turn off xticks and ticks
            axes[-1][-1].tick_params(labelcolor='k', top='off', bottom='off', left='off', right='off')
            
            # Set spine visibility depending on whether the axis is at on the right and/or bottom of the grid
            axes[-1][-1].spines['top'].set_visible(False)
            axes[-1][-1].spines['right'].set_visible(False)
            axes[-1][-1].spines['left'].set_visible(False)
            axes[-1][-1].spines['bottom'].set_visible(False)
            
            # Set tick visibility
            if not axis_column_ind:
                axes[-1][-1].tick_params(axis='y',which='both',left='off',right='off',labelleft='on')
            else:
                axes[-1][-1].tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
                
            if axis_row_ind == number_features-1:
                axes[-1][-1].tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='on')
            else:
                axes[-1][-1].tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
                
            # Set axis labels
            if not axis_column_ind:
                axes[-1][-1].set_ylabel(row_feature,color=text_and_line_color,size=text_font_size,labelpad=25) #labelpad=25,
            
            if axis_row_ind == number_features-1:
                axes[-1][-1].set_xlabel(col_feature,color=text_and_line_color,size=text_font_size,labelpad=25) #,labelpad=25

            # Set axis labels if on left and/or bottom edges
            if not axis_column_ind:
                axes[-1][-1].set_y_label = row_feature
            
            # Populate axes depending on position
            if axis_row_ind == axis_column_ind: # Diagonals
                # Use first standard color
                color_val = get_color_val(0,1)
                
                if row_type == 'numerical':
                    # Plot full histogram of numerical values
                    x = df[row_feature].values
                    
                    axes[-1][-1].hist(x,alpha=bar_alpha,bins=20,color=color_val)
                elif row_type == 'category':
                    # Plot horizontal bar chart that reflects values of the categorical feature
                    unique_feature_values = list(df[row_feature].value_counts().index.values)
                    unique_feature_value_counts = df[row_feature].value_counts().values
                    
                    ind = np.arange(len(unique_feature_values))    # the y locations for the groups

                    bars = axes[-1][-1].barh(ind,unique_feature_value_counts,color=color_val,alpha=bar_alpha,align='center')
                    
                    # Set each bar as the color of the category for reference
                    for bar_ind,bar in enumerate(bars): 
                        bar.set_color(get_color_val(bar_ind,ind.shape[0]))
                        bar.set_label(unique_feature_values[bar_ind])
                        
                    # Set bar labels if at edge
                    if not axis_column_ind:
                        axes[-1][-1].set_yticks(ind)
                        axes[-1][-1].set_yticklabels(unique_feature_values,size=label_size)
                        
                        
            elif row_type == 'category' and col_type == 'numerical':
                # Figure out unique category values
                unique_feature_values = list(df[row_feature].value_counts().index.values)
                
                bins = np.linspace(df[col_feature].min(),df[col_feature].max(),num_bars)
                
                for unique_feature_value_ind,unique_feature_value in enumerate(unique_feature_values):
                    colorVal = get_color_val(unique_feature_value_ind,len(unique_feature_values))
                    
                    data = df[col_feature][df[row_feature]==unique_feature_value].values

                    axes[-1][-1].hist(data,alpha=bar_alpha,bins=bins,label=unique_feature_values,color=colorVal)
                    
                # loop through all patch objects and collect ones at same x
                numLines = len(unique_feature_values)

                # Create dictionary of lists containg patch objects at the same x-postion
                patchDict = {}
                for patch in axes[-1][-1].patches:
                    patchXPosition = patch.get_x()
                    
                    # Initialize x-position list in patch dictionary if not present
                    if patchXPosition not in patchDict:
                        patchDict[patchXPosition] = []
                    
                    # Add dictionary object
                    patchDict[patchXPosition].append(patch)
                
                # Custom sort function, in reverse order of height
                def yHeightSort(i,j):
                    if j.get_height() > i.get_height():
                        return 1
                    else:
                        return -1
                
                # loop through sort assign z-order based on sort
                for x_pos, patches in patchDict.iteritems():
                    if len(patches) == 1:
                        continue
                    patches.sort(cmp=yHeightSort)
                    [patch.set_zorder(patches.index(patch)+numLines) for patch in patches]                
                
            elif row_type == 'category' and col_type == 'category':
                # Get row/column feature value counts
                unique_row_feature_values = df[row_feature].value_counts().sort_index().index.values
                
                col_feature_value_counts = df[col_feature].value_counts().sort_index()
                unique_col_feature_values = col_feature_value_counts.index.values
                
                # Derive concatenated dataframe
                split_data = {}
                split_data = {str(unique_col_feature_value): df[row_feature][df[col_feature]==unique_col_feature_value].value_counts() \
                              for unique_col_feature_value in unique_col_feature_values}

                # Combine data
                all_value_counts = pd.concat(split_data, axis=1).reset_index().sort_values(by=['index'])
                
                # Fill N/A count values with zero
                all_value_counts.fillna(0,inplace=True)
                                
                # Initalize value for bottom bar for stacked bar charts
                bottom_bar_buffer = np.zeros(len(all_value_counts))
                
                bar_objects = []
                for unique_col_feature_value_ind,unique_col_feature_value in enumerate(unique_col_feature_values):                                
                #for unique_row_feature_value_ind,unique_row_feature_value in enumerate(unique_row_feature_values):
                    # Calculate color for bars
                    colorVal = get_color_val(unique_col_feature_value_ind,len(unique_col_feature_values))

                    # Get data for current col_feature value and column_feature
                    data = all_value_counts[str(unique_col_feature_value)]

                    if unique_col_feature_value_ind:
                        previous_feature_value = unique_col_feature_values[unique_col_feature_value_ind-1]
                        
                        bottom_bar_buffer = bottom_bar_buffer + all_value_counts[str(previous_feature_value)]
                    
                    # Calculate bar positions
                    ind = np.arange(len(all_value_counts))    # the x locations for the groups

                    # Set bottom plot keyword arguments
                    plot_kwargs = {
                        'color': colorVal,
                        'left': bottom_bar_buffer,
                        'align': 'center'
                    }
                    bar_objects.append(axes[-1][-1].barh(ind,data,**plot_kwargs))
                    
                # Set bar labels if at edge
                if not axis_column_ind:
                    axes[-1][-1].set_yticks(ind)
                    axes[-1][-1].set_yticklabels(unique_row_feature_values,size=label_size)
                                
            elif row_type == 'numerical' and col_type == 'numerical':
                x = df[row_feature].values
                y = df[col_feature].values
                
                color_val = get_color_val(0,1)
                
                axes[-1][-1].plot(x,y,linestyle='None',marker='o',markerfacecolor=color_val,markersize=2)

            else:
                pass
            
            
    plt.tight_layout()
    
    ###############################################################################################################
    #g = sns.PairGrid(data,vars=plot_vars,hue=hue_feature,
    #                 size=fig_size,aspect=fig_aspect,palette=palette)
    ###############################################################################################################
    #
    #class PairGrid(Grid):
    #    """Subplot grid for plotting pairwise relationships in a dataset."""
    #
    #    def __init__(self, data, hue=None, hue_order=None, palette=None,
    #                 hue_kws=None, vars=None, x_vars=None, y_vars=None,
    #                 diag_sharey=True, size=2.5, aspect=1,
    #                 despine=True, dropna=True):
    #
    #        # Sort out the variables that define the grid
    #        if vars is not None:
    #            x_vars = list(vars)
    #            y_vars = list(vars)
    #        elif (x_vars is not None) or (y_vars is not None):
    #            if (x_vars is None) or (y_vars is None):
    #                raise ValueError("Must specify `x_vars` and `y_vars`")
    #        else:
    #            numeric_cols = self._find_numeric_cols(data)
    #            x_vars = numeric_cols
    #            y_vars = numeric_cols
    #
    #        if np.isscalar(x_vars):
    #            x_vars = [x_vars]
    #        if np.isscalar(y_vars):
    #            y_vars = [y_vars]
    #
    #        self.x_vars = list(x_vars)
    #        self.y_vars = list(y_vars)
    #        self.square_grid = self.x_vars == self.y_vars
    #
    #        # Create the figure and the array of subplots
    #        figsize = len(x_vars) * size * aspect, len(y_vars) * size
    #
    #        fig, axes = plt.subplots(len(y_vars), len(x_vars),
    #                                 figsize=figsize,
    #                                 sharex="col", sharey="row",
    #                                 squeeze=False)
    #
    #        self.fig = fig
    #        self.axes = axes
    #        self.data = data
    #
    #        # Save what we are going to do with the diagonal
    #        self.diag_sharey = diag_sharey
    #        self.diag_axes = None
    #
    #        # Label the axes
    #        self._add_axis_labels()
    #
    #        # Sort out the hue variable
    #        self._hue_var = hue
    #        if hue is None:
    #            self.hue_names = ["_nolegend_"]
    #            self.hue_vals = pd.Series(["_nolegend_"] * len(data),
    #                                      index=data.index)
    #        else:
    #            hue_names = utils.categorical_order(data[hue], hue_order)
    #            if dropna:
    #                # Filter NA from the list of unique hue names
    #                hue_names = list(filter(pd.notnull, hue_names))
    #            self.hue_names = hue_names
    #            self.hue_vals = data[hue]
    #
    #        # Additional dict of kwarg -> list of values for mapping the hue var
    #        self.hue_kws = hue_kws if hue_kws is not None else {}
    #
    #        self.palette = self._get_palette(data, hue, hue_order, palette)
    #        self._legend_data = {}
    #
    #        # Make the plot look nice
    #        if despine:
    #            utils.despine(fig=fig)
    #        fig.tight_layout()
    #
    #
    ## Add histograms to diagonals
    #g.map_diag(plt.hist,alpha=bar_alpha,bins=num_bars)
    #
    ## Plot median lines to histograms in diagonals if specified
    #def plot_median(x,color=[],label=[]): 
    #    plt.axvline(x.median(),alpha=1.0,label=label,color=color)
    #if plot_medians:        
    #    g.map_diag(plot_median)
    #
    #g.map_offdiag(plt.scatter,alpha=scatter_alpha,s=scatter_size)
    #
    ## Add legend if there is a hue feature
    #if hue_feature:
    #    g.add_legend();    
    #
    
    ##############################################################################################################    
            


#                     N = 5
#                     menMeans = (20, 35, 30, 35, 27)
#                     womenMeans = (25, 32, 34, 20, 25)
#                     menStd = (2, 3, 4, 1, 2)
#                     womenStd = (3, 5, 2, 3, 3)
#                     ind = np.arange(N)    # the x locations for the groups
#                     width = 0.35       # the width of the bars: can also be len(x) sequence

                    
#                     p1 = plt.bar(ind, menMeans, width, color='r', yerr=menStd)
#                     p2 = plt.bar(ind, womenMeans, width, color='y',
#                                  bottom=menMeans, yerr=womenStd)

#                     plt.ylabel('Scores')
#                     plt.title('Scores by group and gender')
#                     plt.xticks(ind + width/2., ('G1', 'G2', 'G3', 'G4', 'G5'))
#                     plt.yticks(np.arange(0, 81, 10))
#                     plt.legend((p1[0], p2[0]), ('Men', 'Women'))

#                     plt.show()
                    
                
                

                
#                 plt.hist,alpha=bar_alpha,bins=num_bars
#                 colors = ['red', 'tan', 'lime']

#                 ax0.hist(x, n_bins, normed=1, histtype='bar', color=colors, label=colors)
#                 ax0.hist(x, n_bins, normed=1, histtype='bar', color=colors, label=colors)
      
    
    # Iterate through each block
        # Figure out what type of plot to make based on features
        # Create plot for that position
        # Add legend (possibly in the plot or far to the right or left)


def continuous_pair_grid_vs_label(df,plot_vars=[],hue_feature=[],scatter_alpha=0.2,
                                  bar_alpha=0.3,num_bars=20,scatter_size=45,
                                  palette=['grey','orange','red'],fig_size=6,fig_aspect=1,
                                  filter_feature=[],filter_feature_value=[],plot_medians=False,
                                  plot_means=False,large_text_size=20,small_text_size=16):

    """
    Graphs scatter plot of each feature versus the other in a grid (Seaborn PairGrid plot) 
    with the diagonals being the distribution of the corresponding feature. Sets color of 
    each data point based on user-designated feature (hue_feature).
    
    Requires at least a list of continuous features (plot_vars) and the hue_feature. 
    """
    # Change overall font size
    matplotlib.rcParams['font.size'] = large_text_size

    # Use all features if not explicitly provided by user
    if not plot_vars:
        plot_vars = list(df.columns)
    
    # Filter data if desired
    if filter_feature and filter_feature_value:
        data = df[df[filter_feature]==filter_feature_value]
    elif not filter_feature and not filter_feature_value:
        data = df
    else:
        raise NameError('There must be BOTH a filter_feature and its corresponding filter_feature_value')
    
    # Set hue_feature to None if not provided by user
    if not hue_feature:
        hue_feature = None
    
    # Graph pair grid
    g = sns.PairGrid(data,vars=plot_vars,hue=hue_feature,
                     size=fig_size,aspect=fig_aspect,palette=palette) 
    
    # Add histograms to diagonals
    g.map_diag(plt.hist,alpha=bar_alpha,bins=num_bars)
    
    # Plot median lines to histograms in diagonals if specified
    def plot_median(x,color=[],label=[]): 
        plt.axvline(x.median(),alpha=1.0,label=label,color=color)
        
    # Plot median lines to histograms in diagonals if specified
    def plot_mean(x,color=[],label=[]): 
        plt.axvline(x.mean(),alpha=0.5,label=label,color=color,linestyle='--')
        
    if plot_medians:        
        g.map_diag(plot_median)
    if plot_means:
        g.map_diag(plot_mean)
    
    g.map_offdiag(plt.scatter,alpha=scatter_alpha,s=scatter_size)
    
    # Add legend if there is a hue feature
    if hue_feature:
        g.add_legend();
        
def plot_label_vs_continuous(df,input_feature,output_label_feature,output_labels=[],
                            colors=[],alpha=0.6,figure_size=[12,6],title=[],y_label=[],
                            num_bars=20,plot_medians=True,plot_quantiles=False):
    """
    Plots the distributions of the input_feature for each output_label_feature value
    """
    # Form automatic title if not provided
    if not title:
        title = '%s distributions by %s labels'%(input_feature,output_label_feature)
        
    # Set font attributes
    large_text_size = 14
    small_text_size = 12
    
    # Set axis/legend labels
    if y_label:
        y_label = output_label_feature
    else:
        y_label = 'Frequency'
    x_label = input_feature
        
    # Obtain unique output label feature values
    unique_output_label_feature_values = df[output_label_feature].value_counts().sort_index().index.values

    # Set bin bounds for cleaner plots
    bins = np.linspace(df[input_feature].min(),df[input_feature].max(),num_bars)
    
    # Plot data
    cmap = matplotlib.cm.autumn
    axes = []
    lines = []
    for unique_output_label_feature_value_ind,unique_output_label_feature_value in enumerate(unique_output_label_feature_values):
        # Obtain current data
        current_distribution = df[input_feature][df[output_label_feature]==unique_output_label_feature_value]
        
        # Set series color
        if colors:
            series_color = colors[unique_output_label_feature_value_ind]
        else:
            series_color = cmap(unique_output_label_feature_value_ind)
                    
        # Plot histogram and save axis
        axes.append(current_distribution.plot(kind='hist',color=series_color,
                                              alpha=alpha,figsize=figure_size,bins=bins))
        
    # Obtain data handles for use in legend
    h,_ = axes[-1].get_legend_handles_labels()
            
    # Plot median lines if desired
    if plot_medians:
        for unique_output_label_feature_value_ind,unique_output_label_feature_value in enumerate(unique_output_label_feature_values):
            # Obtain current data
            current_distribution = df[input_feature][df[output_label_feature]==unique_output_label_feature_value]
            
            # Set series color
            if colors:
                series_color = colors[unique_output_label_feature_value_ind]
            else:
                series_color = cmap(unique_output_label_feature_value_ind)
                
            # Plot median lines to histograms in diagonals if specified
            axes[-1].axvline(current_distribution.median(),alpha=0.9,color=series_color)

    # Plot 0, 25, 75, and 100% quartiles if desired
    if plot_quantiles:
        for unique_output_label_feature_value_ind,unique_output_label_feature_value in enumerate(unique_output_label_feature_values):
            # Obtain current data
            current_distribution = df[input_feature][df[output_label_feature]==unique_output_label_feature_value]
            
            # Set series color
            if colors:
                series_color = colors[unique_output_label_feature_value_ind]
            else:
                series_color = cmap(unique_output_label_feature_value_ind)
                
            # Plot median lines to histograms in diagonals if specified
            axes[-1].axvline(current_distribution.quantile(0.25),alpha=0.5,label=unique_output_label_feature_value,color=series_color,linestyle='--')
            axes[-1].axvline(current_distribution.quantile(0.75),alpha=0.5,label=unique_output_label_feature_value,color=series_color,linestyle='--')
            axes[-1].axvline(current_distribution.quantile(0.0),alpha=0.25,label=unique_output_label_feature_value,color=series_color,linestyle='--')
            axes[-1].axvline(current_distribution.quantile(1.0),alpha=0.25,label=unique_output_label_feature_value,color=series_color,linestyle='--')
                
    # Set title, x and y labels, and legend values
    plt.title(title,size=large_text_size)

    # Place x- and y-labels
    plt.xlabel(x_label,size=small_text_size)
    plt.ylabel(y_label,size=small_text_size)
    
    # Place legend
    if output_labels:
        unique_output_label_feature_values
        legend_labels = output_labels
        plt.legend(h,unique_output_label_feature_values,loc='center left',bbox_to_anchor=(1, 0.5))
        #plt.legend(legend_labels,loc='center left',bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(loc='right',bbox_to_anchor=(1, 0.5))

    # Modify plot limits so last gridline visible
    ax = axes[0]
    yticks, yticklabels = plt.yticks()
    ymin = yticks[0] #(3*xticks[0] - xticks[1])/2.
    ymax = 1.1*(3*yticks[-1]-yticks[-2])/2.
    ax.set_ylim(ymin, ymax)    

    # Set left frame attributes    
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['bottom'].set_color('gray')
    
    # Remove all but bottom frame line    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add grid
    ax.yaxis.grid(True,linestyle='--',linewidth=1)
    
    # Place smallest bars in front
    ## loop through all patch objects and collect ones at same x
    
    ## Obtain number of unique data series
    num_lines = len(unique_output_label_feature_values)
    
    ## Create dictionary of lists containg patch objects at the same x-postion
    patch_dict = {}
    for patch in ax.patches:
        # Get current patch position
        patch_x_position = patch.get_x()
        
        # Initialize x-position list in patch dictionary if not present
        if patch_x_position not in patch_dict:
            patch_dict[patch_x_position] = []
        
        # Add dictionary object
        patch_dict[patch_x_position].append(patch)
    
    ## loop through sort assign z-order based on sort
    for x_pos, patches in patch_dict.iteritems():
        # Check that there is more than one patch
        if len(patches) == 1:
            continue
        
        # Sort patches
        patches.sort(cmp=patch_height_sort)
        
        # Set order of patches
        for patch in patches:
            patch.set_zorder(patches.index(patch)+num_lines) 
            
    # Show plot
    plt.show()        
    

def patch_height_sort(patch_one,patch_two):
    """
    Returns 1 flag if second patch higher
    """
    if patch_two.get_height() > patch_one.get_height():
        return 1
    else:
        return -1

    
def plot_label_versus_label(df,input_label_feature,output_label_feature,output_labels=[],
                            colors=[],alpha=0.6,figure_size=[12,6],x_label=[],title=[]):
    # Form automatic title if not provided
    if not title:
        title = 'Proportations of '+output_label_feature+' category within '+input_label_feature+' populations'

    # Set font attributes
    large_text_size = 14
    small_text_size = 12

    # Set axis/legend labels
    y_label = input_label_feature
    if not x_label:
        x_label = 'Frequency'
        
    # Form pivot table between input and output label features
    label_by_label = df[[input_label_feature,output_label_feature]].pivot_table(columns=[output_label_feature],index=[input_label_feature],aggfunc=len)

    # Plot pivot table as stacked column chart
    ax = label_by_label.plot(kind='barh',stacked=True,color=colors,alpha=alpha,figsize=figure_size)

    # Set title, x and y labels, and legend values
    plt.title(title,size=large_text_size)

    plt.xlabel(x_label,size=small_text_size)
    plt.ylabel(y_label,size=small_text_size)

    if output_labels:
        legend_labels = output_labels
        plt.legend(legend_labels,loc='right',bbox_to_anchor=(1.05, 0.5))
    else:
        plt.legend(loc='right',bbox_to_anchor=(1.05, 0.5))

    # Modify plot limits so last gridline visible
    xticks, xticklabels = plt.xticks()
    xmin = xticks[0] #(3*xticks[0] - xticks[1])/2.
    xmax = (3*xticks[-1] - xticks[-2])/2.
    ax.set_xlim(xmin, xmax)    

    # Set left frame attributes    
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['left'].set_color('gray')
    
    # Remove all but frame line
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add grid
    ax.xaxis.grid(True,linestyle='--',linewidth=1)
    
    # Hide the right and top spines        
    plt.show()