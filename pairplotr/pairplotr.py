import seaborn as sns
sns.set(style="white",color_codes=True)
import inspect

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pandas as pd

def compare_data(df,plot_vars=[],bar_alpha=0.85,
                 bins=20,fig_size=16,fig_aspect=1,marker_size=2,
                 marker_alpha=0.5,scatter_plot_filter=None,zoom=[],
                 plot_medians=True):
    """
    Outputs a pairplot given a Pandas dataframe with these formats for Row feature|Column feature
    combinations in either on- or off-diagonal cells: 
    
    On-diagonal:        
        Categorical|Categorical:    Value counts of feature values ordered by ascending value count and
                                    colored by feature values
        Numerical|Numerical:        Histogram of feature w/ no coloring (or by desired label)
    Off-diagonal:
        Categorical|Categorical:    Stacked value count of row feature values colored by column feature
                                    values
        Categorical|Numerical:      Histograms of column feature for each row feature value colored by
                                    row feature value
        Numerical|Numerical:        Scatter plot of row feature values vs column feature values w/ no
                                    coloring (or by desired label)
    """
    # Initialize figure settings
    figure_parameters = {
        'row_count': [],
        'col_count': [],
        'fig_height': [],
        'fig_width': []
    }
    
    # Encode plot type keyword arguments
    plot_kwargs = {
        'scatter': {
            'linestyle': 'None',
            'marker': 'o',
            'markersize': marker_size,
            'alpha': marker_alpha
        },
        'histogram': {
            'alpha': bar_alpha
        },
        'bar': {
            'alpha': bar_alpha,
            'align': 'center'
        }
    }
    
    # Set text and line color
    grayLevel = 0.6
    text_and_line_color = (0.0,0.0,0.0,grayLevel)
    
    # Create axis keyword arguments
    axis_kwargs = {}

    # Set no x-tick, x-tick label, or y-tick generation as default for grid pair plots    
    if not zoom:
        axis_kwargs['xticks'] = []
        axis_kwargs['yticks'] = []
        axis_kwargs['xticklabels'] = []
        
    # Use all features if not explicitly provided by user
    if not plot_vars:
        plot_vars = list(df.columns)

    # Dervive data types from dataframe
    data_types = {}
    for plot_var in plot_vars:    
        if df[plot_var].dtype.name == 'category':
            data_types[plot_var] = 'category' 
        else:
            data_types[plot_var] = 'numerical'
    
    # Get number of features
    feature_count = len(plot_vars)
    
    # Derive orders and colors of each categorical feature value based on ascending value count
    if scatter_plot_filter:
        feature_list = list(set(plot_vars+[scatter_plot_filter])) # Make sure the scatter plot filter is included
    else:
        feature_list = list(plot_vars)
         
    feature_attributes = {}
    for feature in feature_list:
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
            sorted_feature_values = list(sorted_value_count_df.index.values)
            
            # Save feature value counts for later
            feature_attributes[feature]['feature_value_counts'] = sorted_value_count_df.values
            
            # Save feature value order
            feature_attributes[feature]['feature_value_order'] = sorted_feature_values # Ascending results in the colors I want yet not the right order, so I reverse them here
            
            # Get number of feature values
            feature_value_count = len(sorted_feature_values)
            
            # Generate colors for each feature value
            for feature_value_ind,feature_value in enumerate(list(reversed(sorted_feature_values))):
                feature_attributes[feature]['feature_value_colors'][feature_value] = _get_color_val(feature_value_ind,feature_value_count)
          
    # Generate figure settings
    if not zoom:
        figure_parameters['row_count'] = feature_count
        figure_parameters['col_count'] = feature_count
        figure_parameters['fig_height'] = fig_size
        figure_parameters['fig_width'] = fig_size*fig_aspect
    else:
        figure_parameters['row_count'] = 1
        figure_parameters['col_count'] = 1
        figure_parameters['fig_height'] = fig_size
        figure_parameters['fig_width'] = 0.5*fig_size
        
    # Generate figure
    fig = plt.figure(figsize=[figure_parameters['fig_height'],
                              figure_parameters['fig_width']])
    
    # Plot pair grid or single comparsion
    if not zoom:
        # Make sure there are no frames in the pair-grid plot
        axis_kwargs['frame_on'] = False

        # Graph pair grid
        plot_pair_grid(df,fig,plot_vars=plot_vars,data_types=data_types,
                       bar_alpha=bar_alpha,bins=bins,
                       figure_parameters=figure_parameters,
                       feature_attributes=feature_attributes, 
                       scatter_plot_filter=scatter_plot_filter,plot_kwargs=plot_kwargs,
                       axis_kwargs=axis_kwargs,text_and_line_color=text_and_line_color)
    else:
        # Plot single graph
        _plot_single_comparison(df,fig,features=zoom,data_types=data_types,
                               figure_parameters=figure_parameters,
                               feature_attributes=feature_attributes,plot_kwargs=plot_kwargs,
                               axis_kwargs=axis_kwargs,text_and_line_color=text_and_line_color,
                               plot_medians=plot_medians,bins=bins)
        
def _plot_single_comparison(df,fig,features=[],data_types={},figure_parameters={},
                           feature_attributes={},plot_kwargs={},axis_kwargs={},text_and_line_color=(),
                           plot_medians=True,bins=20):
    """
    Plots a single cell of the pairgrid plot given a Pandas dataframe (df), a list of the row-
    and column-features (features), a dictionary of feature attributes (feature_attributes)
    containing the desired ordering/colors of each value for a given feature, a dictionary
    of the data types (e.g. 'category' or 'numerical') for at least the provided features
    (data_types), and a dictionary of figure parameters that includes the number of rows/columns
    and the figure dimensions. This method is meant to be used by the compare_data() method.
    """
    # Check input types and lengths
    if not features or len(features) != 2 or type(features) is not list:
        raise Exception('Keyword argument, features, must be present and \
                        be a list of only 2 features.')
        
    if not data_types:
        raise Exception("A dictionary of data types ('category' or 'numerical'\
                        must be present as the data_types keyword argument.")
        
    # Get row and column features and types
    row_feature = features[0]
    col_feature = features[1]
    
    # Check for their presence in the data types dictionary
    if row_feature in data_types:
        row_type = data_types[row_feature]
    else:
        raise Exception('Feature %s is not in the data_types keyword \
                        argument'%(row_feature))
    
    if col_feature in data_types:
        col_type = data_types[col_feature]
    else:
        raise Exception('Feature %s is not in the data_types keyword \
                        argument'%(col_feature))

    # Get axis    
    ax = fig.add_subplot(1,1,1,**axis_kwargs)
    
    # Choose plot type and plot
    if row_type == 'category' and col_type == 'numerical':
        unique_row_values = feature_attributes[row_feature]['feature_value_order']
        
        colors = feature_attributes[row_feature]['feature_value_colors']
        
        plot_label_vs_continuous(df,col_feature,row_feature,
                                 output_labels=list(unique_row_values),
                                 colors=colors,alpha=0.6,figure_parameters=figure_parameters,
                                 title=[],y_label=[],bins=bins,
                                 plot_medians=plot_medians,plot_quantiles=False,text_and_line_color=text_and_line_color)
        
    elif row_type == 'numerical' and col_type == 'category':
        raise Exception('The first feature of the zoom keyword argument is numerical and the 2nd is \
                        There are no current plans to support this type of plot.')
    elif row_type == 'category' and col_type == 'category':
        unique_col_values = feature_attributes[col_feature]['feature_value_order']
        
        colors = [feature_attributes[col_feature]['feature_value_colors'][col_value] for col_value in unique_col_values]
        
        plot_label_versus_label(df,ax,features[0],features[1],output_labels=list(unique_col_values),
                            colors=colors,alpha=0.6,figure_parameters=figure_parameters,x_label=[],title=[],
                            feature_attributes=feature_attributes,plot_kwargs=plot_kwargs,bar_edge_color='w',
                            text_and_line_color=text_and_line_color)
    elif row_type == 'numerical' and col_type == 'numerical':
        raise NameError("For the moment, numerical vs numerical can't be plotted")
    else:
        raise NameError('Logic error involving single plot and feature types')
        
def plot_pair_grid(df,fig,plot_vars=[],data_types=[],bar_alpha=0.85,
                   bins=20,dpi=[],figure_parameters=[], #fig_size=12,fig_aspect=1,
                   scatter_plot_filter=None,feature_attributes={},plot_kwargs={},axis_kwargs={},
                   text_and_line_color=()):    
    
    # Move first categorical feature to front column in front of a first numerical feature
    feature_types = [data_types[feature] for feature in plot_vars]
    if 'category' in feature_types and feature_types[0]=='numerical':
        # Pop out first categorical variable
        feature_to_move = plot_vars.pop(feature_types.index('category'))
        
        # Issue warning to user
        print "WARNING: Due to a y-tick label bug I can't seem to overcome, the first continuous " \
              "feature encountered in either the dataframe column list or the user-specified list "\
              "of features keyword argument, plot_vars, %s has been moved to the front of the list "\
              "because the first feature, %s, is numerical.\n" %(feature_to_move,plot_vars[0])
        
        # Move first categorical feature to the front of the list of features
        plot_vars.insert(0,feature_to_move)
        
    # Count number of features
    number_features = len(plot_vars)
    
    # Raise error if the desired feature to color the scatter plots isn't found and/or is not categorical
    if scatter_plot_filter:
        if scatter_plot_filter not in df.columns:
            raise Exception("The scatter_plot_filter keyword argument, %s, is not one of the " \
                            "features."%(scatter_plot_filter))
        
        if data_types[scatter_plot_filter] != 'category':
            raise Exception("The scatter_plot_filter keyword argument, %s, is not defined as a \
                            category in the data_types keyword argument. Only categorical \
                            features can be used to color scatter plots."%(scatter_plot_filter))
        
    # Set default text font, color, and size
    text_family = 'sans-serif'
    text_font = 'Helvetica Neue Light'
    text_font_size = 12
    tick_label_size = text_font_size-3
    
    text_properties = {
        'tick_labels': {
            'family': 'sans-serif', 
            'weight': 'normal',
            'size': tick_label_size
        }
    }
    
    # Set padding of axis labels so they don't overlap with tick-labels
    label_padding = 0.65
    
    # Set bar parameters
    bar_edge_color = 'w'
    
    # Set bar_edge_color as default if provided by user
    if bar_edge_color:
        plot_kwargs['histogram']['edgecolor'] = bar_edge_color
        plot_kwargs['bar']['edgecolor'] = bar_edge_color
    
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
            
            # Determine if this is a diagonal, left-edge, and/or bottom-edge grid cell
            diagonal_flag = False
            left_edge_flag = False
            bottom_edge_flag = False
            if axis_row_ind == axis_column_ind:
                diagonal_flag = True
            
            if axis_column_ind == 0:
                left_edge_flag = True
                
            if axis_row_ind == number_features-1:
                bottom_edge_flag = True

            # Determine plot type
            if row_type == 'numerical' and col_type == 'numerical':
                if diagonal_flag:
                    plot_type = 'histogram'
                else:
                    plot_type = 'scatter'
            elif row_type == 'category' and col_type == 'numerical':
                plot_type = 'histogram'
            elif row_type == 'category' and col_type == 'category':
                plot_type = 'bar'
            elif row_type == 'numerical' and col_type == 'category':
                plot_type = None
            else:
                raise Exception("Logic error invovling plot types encountered.")
            
            # Set axis index
            axis_ind = axis_row_ind*number_features+axis_column_ind+1

            # Add axis subplot
            if plot_type or left_edge_flag or bottom_edge_flag:
                ax = fig.add_subplot(figure_parameters['row_count'],
                                     figure_parameters['col_count'],
                                     axis_ind,**axis_kwargs)
                                    
            # Generate plot in axis
            graph_subplot(ax,df,col_feature,row_feature,feature_attributes,
                          plot_type=plot_type,scatter_plot_filter=scatter_plot_filter,
                          plot_kwargs=plot_kwargs,diagonal_flag=diagonal_flag,
                          bins=bins,bar_edge_color=bar_edge_color,
                          left_edge_flag=left_edge_flag,tick_label_size=tick_label_size,
                          text_and_line_color=text_and_line_color)
                        
            # Set y- and x-axis labels
            if left_edge_flag:
                ax.set_ylabel(row_feature,color=text_and_line_color,size=text_font_size)
                ax.get_yaxis().set_label_coords(-label_padding,0.5)
            if bottom_edge_flag:                
                ax.set_xlabel(col_feature,color=text_and_line_color,size=text_font_size) 
    
            # Set y-axis tick labels if on left 
            if left_edge_flag:
                ax.set_yticklabels(ax.get_yticklabels(),text_properties['tick_labels'])
                                    
def graph_subplot(ax,df,col_feature,row_feature,feature_attributes,
                  plot_type=[],plot_kwargs=[],scatter_plot_filter=[],
                  diagonal_flag=[],bins=20,bar_edge_color=[],
                  left_edge_flag=[],tick_label_size=[],text_and_line_color=()):
    """
    Plots subplot given the axis object, dataframe, row- and column- feature names,    
    """
    if plot_type == 'scatter':
        # Get data
        x = df[col_feature].values
        y = df[row_feature].values
        
        # Plot scatter plot either labeled or unlabeled
        if not scatter_plot_filter:
            # Pick color
            plot_kwargs['scatter']['markerfacecolor'] = _get_color_val(0,1)
            
            ax.plot(x,y,**plot_kwargs['scatter'])
        else:
            # Plot separate datasets
            for feature_value in feature_attributes[scatter_plot_filter]['feature_value_order']:
                # Get data
                x = df[col_feature][df[scatter_plot_filter]==feature_value]
                y = df[row_feature][df[scatter_plot_filter]==feature_value]
                
                # Get color
                color = feature_attributes[scatter_plot_filter]['feature_value_colors'][feature_value]
                
                plot_kwargs['scatter']['markerfacecolor'] = color
                
                # Plot scatter plot for current data
                ax.plot(x,y,**plot_kwargs['scatter'])
                
    elif plot_type == 'histogram':
        # Plot histogram of data based on type of plot and whether on- or off-diagonal
        if diagonal_flag:
            # Get data
            x = df[row_feature].values
            
            # Generate bins based on minimum and maximum and number of bars
            bins = np.linspace(df[row_feature].min(),df[row_feature].max(),bins)

            # Pick color
            color = _get_color_val(0,1)

            # Set series-specific parameters                
            ## Set lower value to start bars from
            plot_kwargs[plot_type]['bins'] = bins
            
            ## Set series label
            plot_kwargs[plot_type]['label'] = row_feature
            
            ## Set bar colors
            plot_kwargs[plot_type]['color'] = color
            
            ## Set bar edge color (default is color of current bar unless provided by user)
            if not bar_edge_color:
                plot_kwargs[plot_type]['edgecolor'] = color
            
            # Plot histogram
            ax.hist(x,**plot_kwargs[plot_type])
        else:                    
            # Get unique category values
            unique_row_feature_values = feature_attributes[row_feature]['feature_value_order']
            
            # Generate bins based on minimum and maximum and number of bars
            bins = np.linspace(df[col_feature].min(),df[col_feature].max(),bins)
            
            # Plot a histogram for the column-feature for each row-feature value
            for unique_feature_value in unique_row_feature_values:
                # Obtain color of current histogram
                color = feature_attributes[row_feature]['feature_value_colors'][unique_feature_value]
                
                # Get data for current histogram
                data = df[col_feature][df[row_feature]==unique_feature_value].values
    
                # Set series-specific parameters                
                ## Set lower value to start bars from
                plot_kwargs[plot_type]['bins'] = bins
                
                ## Set series label
                plot_kwargs[plot_type]['label'] = unique_row_feature_values
                
                ## Set bar colors
                plot_kwargs[plot_type]['color'] = color
                
                ## Set bar edge color (default is color of current bar unless provided by user)
                if not bar_edge_color:
                    plot_kwargs[plot_type]['edgecolor'] = color
                
                # Plot current histogram
                ax.hist(data,**plot_kwargs[plot_type])
            
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
            
            # Set series-specific parameters
            ## Set lower value to start bars from
            plot_kwargs[plot_type]['left'] = 0
            
            ## Set bar colors
            plot_kwargs[plot_type]['color'] = color
            
            ## Set bar edge color (default is color of current bar unless provided by user)
            if not bar_edge_color:
                plot_kwargs[plot_type]['edgecolor'] = color
            
            # Draw bar plot
            bars = ax.barh(bar_positions,sorted_row_value_counts,**plot_kwargs[plot_type])

            # Set each bar as the color corresponding to each row feature value 
            for sorted_row_value_ind,sorted_row_value in enumerate(sorted_row_values):
                bar_color = feature_attributes[row_feature]['feature_value_colors'][sorted_row_value]
                
                bars[sorted_row_value_ind].set_color(bar_color)
                
                if not bar_edge_color:
                    bars[sorted_row_value_ind].set_edgecolor(bar_color)
                else:
                    bars[sorted_row_value_ind].set_edgecolor(bar_edge_color)
                
                bars[sorted_row_value_ind].set_label(sorted_row_value)
        else:                    
            # Obtain column feature
            unique_col_feature_values = feature_attributes[col_feature]['feature_value_order']

            # Get the row feature count data for each value of the column feature and sort in descending order  
            split_data = {}
            for unique_col_feature_value in unique_col_feature_values:
                # Find and save data of row feature with current value of column feature,
                # count the number of each row feature value, and sort by the order
                # determined by the total count of each row feature value
                sorted_filtered_data = df[row_feature][df[col_feature]==unique_col_feature_value].value_counts().loc[sorted_row_values]
                
                # Fill N/A values with zero
                sorted_filtered_data.fillna(0,inplace=True)
                
                # Add data to dictionary
                split_data[str(unique_col_feature_value)] = sorted_filtered_data.values
                
            # Initalize value for bottom bar for stacked bar charts
            bottom_bar_buffer = np.zeros(len(sorted_row_values))

            # Plot stacked bars for row feature data corresponding to each column feature value
            for unique_col_feature_value_ind,unique_col_feature_value in enumerate(unique_col_feature_values):                                
                # Get color for bars
                color = feature_attributes[col_feature]['feature_value_colors'][unique_col_feature_value]
                
                # Get data for current col_feature value and column_feature
                data = split_data[str(unique_col_feature_value)]
        
                # Add previous values to current buffer
                if unique_col_feature_value_ind:
                    previous_feature_value = unique_col_feature_values[unique_col_feature_value_ind-1]
                    
                    bottom_bar_buffer = bottom_bar_buffer + split_data[str(previous_feature_value)]                                                
                                        
                # Set series-specific parameters
                ## Set lower value to start bars from
                plot_kwargs[plot_type]['left'] = bottom_bar_buffer
                
                # Set bar colors
                plot_kwargs[plot_type]['color'] = color
                
                ## Set bar edge color (default is color of current bar unless provided by user)
                if not bar_edge_color:
                    plot_kwargs[plot_type]['edgecolor'] = color
                
                # Plot bars corresponding to current series
                ax.barh(bar_positions,data,**plot_kwargs[plot_type])
            
        # Set y-tick positions and labels if against left-side                    
        if left_edge_flag:
            ax.set_yticks(bar_positions)
            ax.set_yticklabels(tick_labels,size=tick_label_size,color=text_and_line_color)
            
def plot_label_vs_continuous(df,input_feature,output_label_feature,output_labels=[],
                            colors=[],alpha=0.6,figure_parameters={},title=[],y_label=[],
                            bins=20,plot_medians=True,plot_quantiles=False,text_and_line_color=()):
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
    bins = np.linspace(df[input_feature].min(),df[input_feature].max(),bins)
    
    # Plot data
    cmap = matplotlib.cm.autumn
    axes = []
    lines = []
    for unique_output_label_feature_value_ind,unique_output_label_feature_value in enumerate(unique_output_label_feature_values):
        # Obtain current data
        current_distribution = df[input_feature][df[output_label_feature]==unique_output_label_feature_value]
        
        # Set series color
        if colors:
            series_color = colors[unique_output_label_feature_value]
        else:
            series_color = cmap(unique_output_label_feature_value_ind)
                    
        # Plot histogram and save axis
        axes.append(current_distribution.plot(kind='hist',color=series_color,
                                              alpha=alpha, 
                                              figsize=[figure_parameters['fig_height'],
                                                       figure_parameters['fig_width']],
                                              bins=bins))
        
    # Obtain data handles for use in legend
    h,_ = axes[-1].get_legend_handles_labels()
            
    # Plot median lines if desired
    if plot_medians:
        for unique_output_label_feature_value_ind,unique_output_label_feature_value in enumerate(unique_output_label_feature_values):
            # Obtain current data
            current_distribution = df[input_feature][df[output_label_feature]==unique_output_label_feature_value]
            
            # Set series color
            if colors:
                series_color = colors[unique_output_label_feature_value]
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
    ax.spines['bottom'].set_color(text_and_line_color)
    
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

def plot_label_versus_label(df,ax,data_feature,hue_feature,output_labels=[],
                            colors=[],alpha=0.6,figure_parameters={},x_label=[],title=[],
                            feature_attributes={},plot_kwargs={},bar_edge_color='w',
                            text_and_line_color=()):
    # Form automatic title if not provided
    if not title:
        title = 'Proportations of '+hue_feature+' category within '+data_feature+' populations'

    # Set plot_type
    plot_type = 'bar'

    # Set font attributes
    large_text_size = 14
    small_text_size = 12
    
    # Set axis/legend labels
    y_label = data_feature
    if not x_label:
        x_label = 'Frequency'
        
    # Form pivot table between input and output label features
    label_by_label = df[[data_feature,hue_feature]].pivot_table(columns=[hue_feature],index=[data_feature],aggfunc=len)

    # Order by specific data feature value 
    label_by_label = label_by_label.loc[feature_attributes[data_feature]['feature_value_order']]
    
    # Fill in N/A values with zero
    label_by_label.fillna(0,inplace=True)

    # Obtain column feature
    unique_col_feature_values = feature_attributes[hue_feature]['feature_value_order']
            
    # Initalize value for bottom bar for stacked bar charts
    sorted_row_values = feature_attributes[data_feature]['feature_value_order']
    bottom_bar_buffer = np.zeros(len(sorted_row_values))
    
    # Set tick-labels
    tick_labels = sorted_row_values
    
    # Set bar and tick-label positions
    bar_positions = np.arange(len(sorted_row_values))

    # Plot stacked bars for row feature data corresponding to each column feature value
    for unique_col_feature_value_ind,unique_col_feature_value in enumerate(unique_col_feature_values):                                
        # Get color for bars
        color = feature_attributes[hue_feature]['feature_value_colors'][unique_col_feature_value]
        
        # Get data for current col_feature value and column_feature
        data = label_by_label[unique_col_feature_value]
        
        # Add previous values to current buffer
        if unique_col_feature_value_ind:
            previous_feature_value = unique_col_feature_values[unique_col_feature_value_ind-1]
            
            bottom_bar_buffer = bottom_bar_buffer + label_by_label[previous_feature_value]
                                
        # Set series-specific parameters
        ## Set lower value to start bars from
        plot_kwargs[plot_type]['left'] = bottom_bar_buffer
        
        # Set bar colors
        plot_kwargs[plot_type]['color'] = color
        
        ## Set bar edge color (default is color of current bar unless provided by user)
        if not bar_edge_color:
            plot_kwargs[plot_type]['edgecolor'] = color
        
        # Plot bars corresponding to current series
        ax.barh(bar_positions,data,**plot_kwargs[plot_type])
        
    # Modify horizontal plot limits so last gridline visible on x-axis
    x_ticks = ax.get_xticks()
    
    x_min = x_ticks[0] #(3*xticks[0] - xticks[1])/2.
    x_max = (3*x_ticks[-1]-x_ticks[-2])/2.
    
    ax.set_xlim(x_min,x_max)
    
    # Modify vertical plot limits so there's a bit of padding on the y-axis
    yticks = ax.get_yticks()
    
    ymin = (3*yticks[0]-yticks[1])/2.
    ymax = (3*yticks[-1]-yticks[-2])/2.
    
    ax.set_ylim(ymin, ymax)

    # Set x-tick label sizes and colors
    for x_tick in ax.xaxis.get_major_ticks():
        x_tick.label.set_fontsize(small_text_size)
        x_tick.label.set_color(text_and_line_color)
        
    # Set y-tick positions and labels
    ax.set_yticks(bar_positions)
    ax.set_yticklabels(tick_labels,size=small_text_size,color=text_and_line_color)
    
    # Set title, x and y labels, and legend values
    ax.set_title(title,size=large_text_size,color=text_and_line_color)

    ax.set_xlabel(x_label,color=text_and_line_color,size=small_text_size)
    ax.set_ylabel(y_label,color=text_and_line_color,size=small_text_size)
    
    if output_labels:
        legend_labels = output_labels
        plt.legend(legend_labels,loc='right',bbox_to_anchor=(1.05, 0.5))
    else:
        plt.legend(loc='right',bbox_to_anchor=(1.05, 0.5))
    
    # Set left frame attributes
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['left'].set_color(text_and_line_color)
    
    # Remove all but frame line
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Add grid
    ax.xaxis.grid(True,linestyle='--',linewidth=1)
    
    # Hide the right and top spines        
    plt.show()
    
def _get_color_val(ind,num_series):
    # Default colormap
    colormap = 'viridis'
    color_map = plt.get_cmap(colormap)
    
    custom_map = ['gray','cyan','orange','magenta','lime','red','purple','blue','yellow','black']
    
    # Choose color
    if num_series > len(custom_map):
        if not ind:
            color = 'gray'
        else:
            if num_series != 1:
                color = color_map(float(ind)/(float(num_series)-1))
            else:
                color = color_map(float(ind)/(float(num_series)))
            
    else:
        color = custom_map[ind]
    
    return color