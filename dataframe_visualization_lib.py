import seaborn as sns
sns.set(style="white",color_codes=True)
import inspect
import matplotlib
import matplotlib.pyplot as plt

def plot_label_versus_label(input_label_feature,output_label_feature,output_labels=[],
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
    
# input_label_feature = 'Sex'
# output_label_feature = 'Survived'
# output_labels = ['Died','Survived']
# colors = ['grey','orange','red']
# x_label = 'Number of passengers'

# plot_label_versus_label(input_label_feature,output_label_feature,output_labels=output_labels,
#                             colors=colors,x_label=x_label)

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