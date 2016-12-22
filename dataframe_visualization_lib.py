import seaborn as sns
sns.set(style="white",color_codes=True)
import inspect
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pandas as pd

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
    
    On-diagonal:        
        Categorical: Value counts of feature values ordered by ascending value count and colored by feature values
        Numerical: Histogram of feature w/ no coloring (or by desired label)
    Off-diagonal:
        Categorical row vs categorical column: Stacked value count of row feature values colored by column feature values
        Categorical row vs numerical column: Histograms of column feature for each row feature value colored by row feature value
        Numerical row vs numerical column: Scatter plot of row feature values vs column feature values w/ no coloring (or by desired label)
        
    Need:   1) An order for each row feature value (by ascending total count)
            2) Colors for those row feature values for when they intersect with a numerical column feature
            3) An order for each column feature value (by ascending total count)
            4) A color for each column feature value
    """
    
    # Use all features if not explicitly provided by user
    if not plot_vars:
        plot_vars = list(df.columns)
        
    # Check that data_types are specified
    if not data_types:
        raise Exception('Dictionary of feature:data types keyword argument, data_types, must be specified.')
    
    # Keep only features specified by the user that are also included in the data types dictionary
    plot_vars = [plot_var for plot_var in plot_vars if plot_var in data_types]
    
    # Count number of features
    number_features = len(plot_vars)
    
    ################## SET FIGURE DEFAULTS ##################
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
    bar_edge_color = None
    
    # Set marker parameters
    marker_size = 2
    
    # Generate figure
    fig = plt.figure(figsize=[fig_size,fig_size*fig_aspect])
    
    # Derive orders and colors of each categorical feature value based on ascending value count
    feature_attributes = {}
    for feature in plot_vars:
        # Get feature type
        feature_type = data_types[feature]
        
        # Initialize new features
        if feature not in feature_attributes:
            feature_attributes[feature] = {
                'feature_value_order': [],
                'feature_value_colors': {},
                'feature_value_counts': {}
            }
        
        # Get feature value order, value counts, and color for each value
        if feature_type == 'category':
            # Get feature value counts and sort in ascending order by count
            sorted_value_count_df = df[feature].value_counts().sort_values(ascending=True)
            
            # Get feature values
            sorted_feature_values = sorted_value_count_df.index.values
            
            # Save feature value counts for later
            feature_attributes[feature]['feature_value_counts'] = sorted_value_count_df.values
            
            # Save feature value order
            feature_attributes[feature]['feature_value_order'] = sorted_feature_values # Ascending results in the colors I want yet not the right order, so I reverse them here
            
            # Get number of feature values
            feature_value_count = len(sorted_feature_values)
            
            # Generate colors for each feature value
            for feature_value_ind,feature_value in enumerate(list(reversed(sorted_feature_values))):
                feature_attributes[feature]['feature_value_colors'][feature_value] = _get_color_val(feature_value_ind,feature_value_count)
                
    # Graph axes
    for axis_row_ind in range(number_features):
        # Get the row feature and its type
        row_feature = plot_vars[axis_row_ind]
        row_type = data_types[row_feature]
        
        # Initialize current row of axes
        for axis_column_ind in range(number_features):
            # Get the column feature and its type
            col_feature = plot_vars[axis_column_ind]
            col_type = data_types[col_feature]
            
            # Determine plot type
            if row_type == 'numerical' and col_type == 'numerical':
                plot_type = 'scatter'
            elif row_type == 'category' and col_type == 'numerical':
                plot_type = 'histogram'
            elif row_type == 'category' and col_type == 'category':
                plot_type = 'bar'
            elif row_type == 'numerical' and col_type == 'category':
                plot_type = None
            else:
                raise Exception("Logic error invovling plot types encountered.")
            
            # Determine if this is a diagonal, left-edge, and/or bottom-edge grid cell
            diagonal_flag = False
            left_edge_flag = False
            bottom_edge_flag = False
            if axis_row_ind == axis_column_ind:
                diagonal_flag = True
                
                # Change plot type to histogram for numerical plots
                if plot_type == 'scatter':
                    plot_type = 'histogram'
                
            if axis_column_ind == 0:
                left_edge_flag = True
                
            if axis_row_ind == number_features-1:
                bottom_edge_flag = True
                
            # Create subplot
            ax = fig.add_subplot(fig_size,fig_size*fig_aspect,axis_row_ind*fig_size*fig_aspect+axis_column_ind+1)
                        
            # Turn off xticks and ticks
            ax.tick_params(labelcolor='k',top='off',bottom='off',left='off',right='off')
            
            # Set spine visibility depending on whether the axis is at on the right and/or bottom of the grid
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            
            # Make only left- and bottom-edge ticks visible
            if left_edge_flag:
                ax.tick_params(axis='y',which='both',left='off',right='off',labelleft='on')
            else:
                ax.tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
                
            if bottom_edge_flag:
                ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
            else:
                ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
                
            # Generate plot
            if plot_type == 'scatter':
                # Get data
                x = df[col_feature].values
                y = df[row_feature].values
                
                # Pick color
                color_val = _get_color_val(0,1)
                
                # Plot scatter plot
                plot_kwargs = {
                    'linestyle': 'None',
                    'marker': 'o',
                    'markerfacecolor': color_val,
                    'markersize': marker_size
                }
                ax.plot(x,y,**plot_kwargs)

            elif plot_type == 'histogram':
                # Plot histogram of data based on type of plot and whether on- or off-diagonal
                if diagonal_flag:
                    # Get data
                    x = df[row_feature].values

                    # Pick color
                    color_val = _get_color_val(0,1)

                    # Plot full histogram of numerical values
                    plot_kwargs = {
                        'alpha': bar_alpha,
                        'bins': 20,
                        'color': color_val,
                        'edgecolor': color
                    }
                    ax.hist(x,**plot_kwargs)
                else:                    
                    # Get unique category values
                    unique_row_feature_values = feature_attributes[row_feature]['feature_value_order']
                    
                    # Generate bins based on minimum and maximum and number of bars
                    bins = np.linspace(df[col_feature].min(),df[col_feature].max(),num_bars)
                    
                    # Plot a histogram for the column-feature for each row-feature value
                    for unique_feature_value_ind,unique_feature_value in enumerate(unique_row_feature_values):
                        # Obtain color of current histogram
                        color = feature_attributes[row_feature]['feature_value_colors'][unique_feature_value]
                        
                        # Get data for current histogram
                        data = df[col_feature][df[row_feature]==unique_feature_value].values
            
                        # Draw current histogram
                        ax.hist(data,alpha=bar_alpha,bins=bins,label=unique_row_feature_values,color=color,edgecolor=color)
                    
                    # Make all bars in multiple overlapping histogram plot visible
                    ## Get number of histograms
                    histogram_count = len(unique_row_feature_values)
            
                    ## Collect Patch objects representing bars at the same position
                    bars = {}
                    for bar in ax.patches:
                        # Get current bar position
                        bar_position = bar.get_x()
                        
                        # Initialize x-position list in bar dictionary if not present
                        if bar_position not in bars:
                            bars[bar_position] = []
                        
                        # Add current bar to collection of bars at that position
                        bars[bar_position].append(bar)
                                                
                    ## Sort bars based on height so smallest is visible
                    for bar_position, bar_group in bars.iteritems():
                        # Sort bars by height order for current bar group at current position
                        if len(bar_group) > 1:
                            # Sort by height
                            bar_group = sorted(bar_group, key=lambda x: x.get_height(),reverse=True)
                            
                            # Set layer position to current order
                            for bar_ind,bar in enumerate(bar_group):
                                bar.set_zorder(bar_ind)
            elif plot_type == 'bar':
                # Get row feature values and counts sorted by ascending counts
                sorted_row_values = feature_attributes[row_feature]['feature_value_order']
                sorted_row_value_counts = feature_attributes[row_feature]['feature_value_counts']
                
                # Set tick-labels
                tick_labels = sorted_row_values
                
                # Set bar and tick-label positions
                bar_positions = np.arange(len(sorted_row_values))
                
                # Draw bar chart based on whether on- or off-diagonal
                if diagonal_flag:
                    # Pick color
                    color = _get_color_val(0,1) # Just pick first color

                    # Draw bars
                    plot_kwargs = {
                        'color': color,
                        'alpha': bar_alpha,
                        'align': 'center',
                        'edgecolor': color
                    }
                    bars = ax.barh(bar_positions,sorted_row_value_counts,**plot_kwargs)

                    # Set each bar as the color corresponding to each row feature value 
                    for sorted_row_value_ind,sorted_row_value in enumerate(sorted_row_values):
                        bar_color = feature_attributes[row_feature]['feature_value_colors'][sorted_row_value]
                        
                        bars[sorted_row_value_ind].set_color(bar_color)
                        
                        bars[sorted_row_value_ind].set_label(sorted_row_value)
                        
                    # Set bar labels if at edge
                    if left_edge_flag:
                        ax.set_yticks(bar_positions)
                        ax.set_yticklabels(tick_labels,size=label_size)
                else:                    
                    # Get individual row values
                    sorted_row_feature_values = feature_attributes[row_feature]['feature_value_order']
                    
                    # Obtain column feature
                    unique_col_feature_values = feature_attributes[col_feature]['feature_value_order']

                    # Get the row feature count data for each value of the column feature and sort in descending order  
                    split_data = {}
                    for unique_col_feature_value in unique_col_feature_values:
                        # Find and save data of row feature with current value of column feature,
                        # count the number of each row feature value, and sort by the order
                        # determined by the total count of each row feature value
                        sorted_filtered_data = df[row_feature][df[col_feature]==unique_col_feature_value].value_counts()[sorted_row_feature_values]
                        
                        # Fill N/A values with zero
                        sorted_filtered_data.fillna(0,inplace=True)
                        
                        # Add data to dictionary
                        split_data[str(unique_col_feature_value)] = sorted_filtered_data.values
                        
                    # Initalize value for bottom bar for stacked bar charts
                    bottom_bar_buffer = np.zeros(len(sorted_row_feature_values))

                    for unique_col_feature_value_ind,unique_col_feature_value in enumerate(unique_col_feature_values):                                
                        # Calculate color for bars
                        color = feature_attributes[col_feature]['feature_value_colors'][unique_col_feature_value]
                        
                        # Get data for current col_feature value and column_feature
                        data = split_data[str(unique_col_feature_value)]
                
                        if unique_col_feature_value_ind:
                            previous_feature_value = unique_col_feature_values[unique_col_feature_value_ind-1]
                            
                            bottom_bar_buffer = bottom_bar_buffer + split_data[str(previous_feature_value)]                                                
                        
                        # Calculate bar positions
                        ind = np.arange(len(sorted_row_feature_values))    # the x locations for the groups
                
                        # Set bottom plot keyword arguments
                        plot_kwargs = {
                            'color': color,
                            'left': bottom_bar_buffer,
                            'align': 'center',
                            'edgecolor': bar_edge_color,
                            'alpha': bar_alpha
                        }
                        ax.barh(ind,data,**plot_kwargs)
                    
                    # Set y-tick positions and labels if against left-side                    
                    if not axis_column_ind:
                        ax.set_yticks(ind)
                        ax.set_yticklabels(sorted_row_values,size=label_size)
                        
            else:
                # Set y-tick positions and labels if against left-side                    
                ax.tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
                         
            # Set y- and x-axis labels
            label_padding = 5
            if not axis_column_ind:
                ax.set_ylabel(row_feature,color=text_and_line_color,size=text_font_size,labelpad=label_padding)
            if axis_row_ind == number_features-1:
                ax.set_xlabel(col_feature,color=text_and_line_color,size=text_font_size,labelpad=label_padding) 

            # Set axis labels if on left and/or bottom edges
            if not axis_column_ind:
                ax.set_y_label = row_feature
            
def _get_color_val(ind,num_series):
    colormap = 'rainbow'
    color_map = plt.get_cmap(colormap)
    
    custom_map = ['grey','cyan','orange','magenta','lime','red','purple','blue','yellow','black']
    
    # Calculate color
    if num_series > len(custom_map):
        if not ind:
            colorVal = 'gray'
        else:
            colorVal = color_map((ind-1)/float(num_series+1))
    else:
        colorVal = custom_map[ind]
    
    return colorVal

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