import seaborn as sns
sns.set(style="white",color_codes=True)
import inspect
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def compare_data(df,plot_vars=[],data_types=[],bar_alpha=0.3,
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
    
    # Form blank pairgrid
    fig, axes = plt.subplots(nrows=number_features, ncols=number_features,figsize=[fig_size,fig_size*fig_aspect])
    for axis_row_ind in range(number_features):
        for axis_column_ind in range(number_features):
            row_feature = plot_vars[axis_row_ind]
            col_feature = plot_vars[axis_column_ind]
            
            row_type = data_types[row_feature]
            col_type = data_types[col_feature]
            
            if axis_row_ind == axis_column_ind:
                if row_type == 'numerical':
                    x = df[row_feature].values
                    y_pos = np.arange(len(x))
                    axes[axis_row_ind][axis_column_ind].hist(x,alpha=bar_alpha,bins=20,
                                                             color=[palette[0]])
                elif row_type == 'category':
                    unique_feature_values = list(df[row_feature].value_counts().index.values)
                    unique_feature_value_counts = df[row_feature].value_counts().values
                    
                    ind = np.arange(len(unique_feature_values))    # the x locations for the groups
                    axes[axis_row_ind][axis_column_ind].bar(ind,unique_feature_value_counts, 
                                                            color=[palette[0]],alpha=bar_alpha)
            elif row_type == 'category' and col_type == 'numerical':
                # Figure out unique category values
                unique_feature_values = list(df[row_feature].value_counts().index.values)
                
                x = [df[col_feature][df[row_feature]==unique_feature_value].values for unique_feature_value in unique_feature_values]
                
                axes[axis_row_ind][axis_column_ind].hist(x,alpha=bar_alpha,bins=20,
                                                         label=unique_feature_values)
                
            elif row_type == 'category' and col_type == 'category':
                pass
            else:
                pass
            
    plt.tight_layout()
            


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
                                  large_text_size=20,small_text_size=16):

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
    if plot_medians:        
        g.map_diag(plot_median)
    
    g.map_offdiag(plt.scatter,alpha=scatter_alpha,s=scatter_size)
    
    # Add legend if there is a hue feature
    if hue_feature:
        g.add_legend();
        
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
    label_by_label.plot(kind='barh',stacked=True,color=colors,alpha=alpha,figsize=figure_size)

    # Set title, x and y labels, and legend values
    plt.title(title,size=large_text_size)

    plt.xlabel(x_label,size=small_text_size)
    plt.ylabel(y_label,size=small_text_size)

    if output_labels:
        legend_labels = output_labels
        plt.legend(legend_labels,loc='best')
    else:
        plt.legend(loc='best')
        
    plt.show()