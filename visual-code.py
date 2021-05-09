# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:33:05 2021

@author: arvind.ramachandran
"""


import holoviews as hv
from bokeh.models.renderers import GraphRenderer
import matplotlib as mpl
from grave import plot_network
from grave.style import use_attributes
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap as Basemap
from pandas import DataFrame
import psycopg2
from geopy.geocoders import Nominatim
import itertools
from matplotlib.colors import rgb2hex, Normalize
from matplotlib.patches import Polygon
from geopy.extra.rate_limiter import RateLimiter
import ssl
import certifi
import geopy.geocoders
from matplotlib.lines import Line2D
from matplotlib.colorbar import ColorbarBase
#import geopandas
import numpy as np
import geopy.distance
from bokeh.sampledata.us_states import data as states

import matplotlib.cm as cm
from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,
                          MultiLine,Div, Row, NodesAndLinkedEdges, Plot, Range1d, TapTool,StaticLayoutProvider)
from bokeh.palettes import Spectral4
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models import HoverTool, TapTool, BoxSelectTool
from bokeh.models.graphs import from_networkx
from bokeh.models.graphs import NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.models import Plot, ColumnDataSource
from bokeh.models.sources import ColumnDataSource, CDSView
mpl.rc('figure', max_open_warning = 0)
hv.extension('bokeh')
cmap = cm.hot
geolocator = Nominatim(user_agent="lv_agent2")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
ctx = ssl.create_default_context(cafile=certifi.where())
geopy.geocoders.options.default_ssl_context = ctx
df = pd.read_csv('USdata.csv',sep=',')
df_src1 = df['src'].unique()
df_src = pd.DataFrame(df_src1, columns = ['src'])
df_tgt1 = df['dest'].unique()
df_tgt= pd.DataFrame(df_tgt1, columns = ['src'])
df_un = pd.concat([df_src, df_tgt]).drop_duplicates().reset_index(drop=True)
df_un.columns = ['src']
df_un['dest'] = df_un['src']
state_names = []
f1 = plt.figure()
m = Basemap(llcrnrlon=-119,llcrnrlat=20,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
shp_info = m.readshapefile('st99_d00','states',drawbounds=True,
                           linewidth=0.45,color='gray')
def geolocate(city=None, country=None):
    '''
    Inputs city and country, or just country. Returns the lat/long coordinates of
    either the city if possible, if not, then returns lat/long of the center of the country.
    '''
   
    # If the city exists,
    if city == 'IL':
        return (39.7837304, -89.3985)
    if city == 'LA':
        return (30.9843, -91.9623)
    if city == 'DC':
        return (38.9072, -77.0369)
    if city == 'IN':
        return (40.2672, -86.1349)
    if city == 'SD':
        return  (43.9695, -99.9018)
    if city == 'VI':
        return  (18.3358,-64.8963)
    if city != None:
        # Try
        try:
            # To geolocate the city and country
            loc = geolocator.geocode(str(city + ',' + country),timeout=10)
            # And return latitude and longitude
            return (loc.latitude , loc.longitude)
           
        # Otherwise
        except:
            # Return missing value
            return np.nan
    # If the city doesn't exist
       
    else:
        # Try
        try:
            # Geolocate the center of the country
            loc = geolocator.geocode(country,timeout=10)
            # And return latitude and longitude
            return (loc.latitude , loc.longitude)
        # Otherwise
        except:
            # Return missing value
            return np.nan
latlon = []
for srce in df_un['src']:
#     ##print(srce)
     t = geolocate(city = srce, country = 'US')
     latlon.append(t)
df_un['latlon'] = latlon
df_un1 = pd.merge(df, df_un, on='src', how='left')
df_un1.rename(columns={'latlon': 'src_lat_lon'}, inplace=True)
df_unf = df_un1.drop(columns=['dest_y'])
df_unf.rename(columns={'dest_x': 'dest'}, inplace=True)
df_un2 = pd.merge(df_unf, df_un, on='dest', how='left')
#df_un2.rename(columns={'latlon': 'dest_lat_lon'})
df_unfn = df_un2.drop(columns=['src_y'])
df_unfn.rename(columns={'src_x': 'src'}, inplace=True)
df_unfn.rename(columns={'latlon': 'dest_lat_lon'}, inplace=True)
df_unfn[['src_lat', 'src_long']] = pd.DataFrame(df_unfn['src_lat_lon'].tolist(), index=df_unfn.index)
df_unfn[['dest_lat', 'dest_long']] = pd.DataFrame(df_unfn['dest_lat_lon'].tolist(), index=df_unfn.index)
df_unfn['tot_le_dist'] = df_unfn.apply(lambda row:geopy.distance.geodesic((row.src_lat,row.src_long),(row.dest_lat,row.dest_long)).miles, axis = 1)
src_nm = [row['src'] for index, row in df_unfn.iterrows()]
dest_nm = [row['dest'] for index, row in df_unfn.iterrows()]
s_lat = [row['src_lat'] for index, row in df_unfn.iterrows()]
s_lon = [row['src_long'] for index, row in df_unfn.iterrows()]
d_lat = [row['dest_lat'] for index, row in df_unfn.iterrows()]
d_lon = [row['dest_long'] for index, row in df_unfn.iterrows()]
G=nx.DiGraph()
G1=nx.Graph()
H=nx.DiGraph()
position = {}
for i in range(0, len(df_unfn)):
    G.add_edge(src_nm[i],dest_nm[i])
deg_centrality = nx.in_degree_centrality(G)
df_cent = pd.DataFrame.from_dict({
    'node': list(deg_centrality.keys()),
    'centrality': list(deg_centrality.values())
})
df_cent = df_cent.groupby('centrality')['node'].apply(list)
dff= pd.DataFrame(df_cent).reset_index()
t_list = {}
df_cent = dff.sort_values('centrality',ascending=False)
t_list_head1 = df_cent['node'].head(2)
t_list_head = list(itertools.chain.from_iterable(t_list_head1))
t_list = t_list_head
df['kd'] = df['dest'].isin(t_list).astype(int)
df2a = df.copy()
df2b = df2a.dropna()
df2c = df2b.sort_values(by='tot_count', ascending=False)
df2c = df2c[df2c.kd != 0]
df2c1 = df2c.groupby(['src', 'dest'])[['tot_count']].agg('sum')
df2c1s = df2c.groupby(['dest'])['ret_count'].agg('sum')
df2c2 = df2c1.reset_index()
df2c3 = df2c2.reset_index()
df2c1st = df2c1s.reset_index()
df2c = df2c3.groupby('dest', as_index=False).apply(lambda x: x.nlargest(10, columns='tot_count')).reset_index(level=0, drop=True)
df2cg = df2c.groupby('dest')['tot_count'].agg('sum')
df2cr1 = df2c3.groupby('dest', as_index=False).apply(lambda x: x.nlargest(10, columns='tot_count')).reset_index(level=0, drop=True)
df2cr1s1 = df2c1st.groupby('dest', as_index=False).apply(lambda x: x.nlargest(10, columns='ret_count')).reset_index(level=0, drop=True)
#print(df2cr1)
df2cg3 = df2cg.to_frame()
df2cr1s = df2cr1s1.reset_index()
df2cgr = df2cr1.groupby(['dest','src'])['tot_count'].agg('sum')
df2cgr3 = df2cgr.to_frame()
df2cgrf = df2cgr3.sort_values(['tot_count'], ascending=False)

df2cgrf1 = df2cr1[['tot_count']]
df2cgrf1s = df2cr1s['ret_count'].to_list()
lts = list(itertools.chain.from_iterable(itertools.repeat(x, 10) for x in df2cgrf1s))
state = []
state2 = []
def geolocate1(city=None, country=None):
    if city == 'LA':
        return  ('Louisiana')
    if city == 'IL':
        return ('Illinois')
    if city == 'DC':
        return ('District of Columbia')
    if city == 'IN':
        return ('Indiana')
    if city == 'LA':
        return  ('Louisiana')
    if city == 'SD':
        return  ('South Dakota')
    if city == 'PR':
        return  ('Puerto Rico')
    if city == 'VI':
        return  ('Virgin Islands')
    else:
        loc = geolocator.geocode(str(city + ',' + country),timeout=10)
for i, row in df2cg3.iterrows():
    tup = geolocate1(city = i, country = 'US')
    state1 = str(tup).split(',')[0]
    state.append(state1)

df2cg3['state'] = state
df2cg4 = df2cg3[['state','tot_count']]
#df2cg4 = df2cg4.set_index('state')
df2cg2 = dict(df2cg4.values.tolist())
colors={}
custom_lines = [Line2D([0], [0], color='orange', lw=4,label = '>1000 miles'),Line2D([0], [0], color='lightseagreen', lw=4, label = '<1000 miles'),
                Line2D([0], [0], marker='o',color='white', markerfacecolor='green', markersize=15, label='Travel State')]
ax = plt.gca() # get current axes instance
ax.set_title('United states  Distribution by state based on Degree Centrality')
ax.legend(handles=custom_lines, loc='lower left')
lt = df2cgrf1['tot_count'].to_list()
t = len(df2cg2)
df_un11 = pd.merge(df2c, df_un, on='src', how='left')
df_un11.rename(columns={'latlon': 'src_lat_lon'}, inplace=True)
df_unf1 = df_un11.drop(columns=['dest_y'])
df_unf1.rename(columns={'dest_x': 'dest'}, inplace=True)
df_un21 = pd.merge(df_unf1, df_un, on='dest', how='left')
#df_un2.rename(columns={'latlon': 'dest_lat_lon'})
df_unfn1 = df_un21.drop(columns=['src_y'])
df_unfn1.rename(columns={'src_x': 'src'}, inplace=True)
df_unfn1.rename(columns={'latlon': 'dest_lat_lon'}, inplace=True)
df_unfn1[['src_lat', 'src_long']] = pd.DataFrame(df_unfn1['src_lat_lon'].tolist(), index=df_unfn1.index)
df_unfn1[['dest_lat', 'dest_long']] = pd.DataFrame(df_unfn1['dest_lat_lon'].tolist(), index=df_unfn1.index)
df_unfn1['distance'] = df_unfn1.apply(lambda row:geopy.distance.geodesic((row.src_lat,row.src_long),(row.dest_lat,row.dest_long)).miles, axis = 1)
df_unfns = df_unfn1.sort_values(by=['src','dest'])
df_unfns_g = df_unfns.groupby(['dest','src']).filter(lambda group: group['distance'] > 1000)
src_nm1 = [row['src'] for index, row in df_unfns.iterrows()]
dest_nm1 = [row['dest'] for index, row in df_unfns.iterrows()]
s_lat1 = [row['src_lat'] for index, row in df_unfns.iterrows()]
s_lon1 = [row['src_long'] for index, row in df_unfns.iterrows()]
d_lat1 = [row['dest_lat'] for index, row in df_unfns.iterrows()]
d_lon1 = [row['dest_long'] for index, row in df_unfns.iterrows()]
distance1  = [row['distance'] for index, row in df_unfns.iterrows()]
G2=nx.OrderedDiGraph()
position1 = {}
srccnt = []
srccnt1 = []
edge_ln = {}

for i in range(0, len(df_unfns)):
    G2.add_edge(src_nm1[i],dest_nm1[i])
    position1[src_nm1[i]] = m(s_lon1[i],s_lat1[i])
    position1[dest_nm1[i]] = m(d_lon1[i],d_lat1[i])
    edge_ln[i] = distance1[i]
weights = []
total = []
edge_colors = []
edge_legends = []
for edge in list(G2.edges()):
    for _,row in df_unfns.iterrows():
        if ((row['src'] == edge[0]) & (row['dest'] == edge[1])) :
            if edge[0] == edge[1] :
                weights.append(0.2)
                total.append(0.2)
            else:
                t = row['distance']
                weights.append(t)
                u = row['tot_count']
                total.append(u)
        else:
            weights.append(0)
            total.append(0)
def dnodes():
    p = []
    for each in G2.nodes():
        p.append(each)
weights = [i for i in weights if i != 0]
weights = [0 if i == 0.2 else i for i in weights ]
totcount = total
total = [i for i in total if i != 0]
total = [0 if i == 0.2 else i*0.00008 for i in total ]

totcount = [i for i in totcount if i != 0]
totcount = [0 if i == 0.2 else i for i in totcount ]
retcount = []
for value in weights:
    if (value > 1000):
        edge_colors.append('orange')
        edge_legends.append('>1000 miles')
    else:
        edge_colors.append('lightseagreen')
        edge_legends.append('<1000 miles')
for node  in G2:
    #print(node)
    if df2cr1s['dest'].str.contains(node).any() == True:
        srccnt1.append(df2cr1s.loc[df2cr1s['dest'] == node, 'ret_count'].iloc[0])
        retcount.append(df2cr1s.loc[df2cr1s['dest'] == node, 'ret_count'].iloc[0])
    else:
        srccnt1.append(13000)
        retcount.append(0)
srccnt = [i*0.00005 for i in srccnt1]
color_map = []
node_legends = []
for node  in G2:
    if node in dest_nm1:
        color_map.append('green')
        node_legends.append('Travel State')
       
    else:
        continue
def from_networkx1(graph, graph_layout, **kwargs):
        '''
        Generate a GraphRenderer from a networkx.Graph object and networkx
        layout function. Any keyword arguments will be passed to the
        layout function.
        Args:
            graph (networkx.Graph) : a networkx graph to render
            layout_function (function) : a networkx layout function
        Returns:
            instance (GraphRenderer)
        .. warning::
            Only two dimensional layouts are currently supported.
        .. warning::
            Node attributes labeled 'index' and edge attributes labeled 'start' or 'end' are ignored.
            If you want to convert these attributes, please re-label them to other names.
        '''

        # inline import to prevent circular imports
        # !!! Comment out once to run on Jupyter Notebook. !!!
        # !!! In fact, the two lines of code are executed. !!!
        # from .models.renderers import GraphRenderer
        # from ..models.graphs import StaticLayoutProvider

        # !!! Lines 27-68 are exactly the same as existing from_networkx. !!!
        node_dict = dict()
        node_attr_keys = [attr_key for node in list(graph.nodes(data=True))
                          for attr_key in node[1].keys()]
        node_attr_keys = list(set(node_attr_keys))

        for attr_key in node_attr_keys:
            node_dict[attr_key] = [node_attr[attr_key] if attr_key in node_attr.keys() else None
                                   for _, node_attr
                                   in graph.nodes(data=True)]
        
        

        if 'index' in node_attr_keys:
            from warnings import warn
            warn("Converting node attributes labeled 'index' are skipped. "
                 "If you want to convert these attributes, please re-label with other names.")

        node_dict['index'] = list(graph.nodes())

        # Convert edge attributes
        edge_dict = dict()
        edge_attr_keys = [attr_key for edge in graph.edges(data=True)
                          for attr_key in edge[2].keys()]
        edge_attr_keys = list(set(edge_attr_keys))

        for attr_key in edge_attr_keys:
            edge_dict[attr_key] = [edge_attr[attr_key] if attr_key in edge_attr.keys() else None
                                   for _, _, edge_attr
                                   in graph.edges(data=True)]
        

        if 'start' in edge_attr_keys or 'end' in edge_attr_keys:
            from warnings import warn
            warn("Converting edge attributes labeled 'start' or 'end' are skipped. "
                 "If you want to convert these attributes, please re-label them with other names.")

        edge_dict['start'] = [x[0] for x in graph.edges()]
        edge_dict['end'] = [x[1] for x in graph.edges()]
        
        
        node_source = ColumnDataSource(data=node_dict)
        edge_source = ColumnDataSource(data=edge_dict)
        
        graph_renderer = GraphRenderer()
        graph_renderer.node_renderer.data_source.data = node_source.data
        graph_renderer.edge_renderer.data_source.data = edge_source.data
        
        #!!! if graph_layout is a function, it is called to calculate a layout. !!!
        if callable(graph_layout):
            graph_layout = graph_layout(graph, **kwargs)
        
        graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
        
        return graph_renderer

def geolocate2(city=None, country=None):
    '''
    Inputs city and country, or just country. Returns the lat/long coordinates of
    either the city if possible, if not, then returns lat/long of the center of the country.
    '''
   
    # If the city exists,
    if city == 'IL':
        return (-89.3985,39.7837304)
    if city == 'LA':
        return (-91.9623,30.9843)
    if city == 'DC':
        return (-77.0369,38.9072)
    if city == 'IN':
        return (-86.1349,40.2672)
    if city == 'SD':
        return  (-99.9018,43.9695)
    if city == 'VI':
        return  (-64.8963,18.3358)
    if city != None:
        # Try
        try:
            # To geolocate the city and country
            loc = geolocator.geocode(str(city + ',' + country),timeout=10)
            # And return latitude and longitude
            return (loc.longitude , loc.latitude)
           
        # Otherwise
        except:
            # Return missing value
            return np.nan
    # If the city doesn't exist
       
    else:
        # Try
        try:
            # Geolocate the center of the country
            loc = geolocator.geocode(country,timeout=10)
            # And return latitude and longitude
            return (loc.longitude , loc.latitude)
        # Otherwise
        except:
            # Return missing value
            return np.nan
latlon2 = []
new_dict = {}

for srce in df_un['src']:
#     ##print(srce)
     t2 = geolocate2(city = srce, country = 'US')
     latlon2.append(t2)
df_un['latlon2'] = latlon2
for node  in G2:
    #print(node)
    if df_un['src'].str.contains(node).any() == True:
        new_dict[node] = df_un.loc[df_un['src'] == node, 'latlon2'].iloc[0]

EXCLUDED = ("AK", "HI", "PR", "GU", "VI", "MP", "AS")
network_graph = from_networkx1(G2, new_dict)
plot = figure(plot_width=1100, plot_height=700,tooltips=None,x_range=Range1d(-130,-60), y_range=Range1d(20,50), title='United states travellers Distribution by state')


nx.draw_networkx(G2,position1,node_color = color_map,node_size=srccnt,
                 with_labels=False,
                 edge_color=edge_colors,
        width=total)

total1 = [i*4 for i in total]
state_xs = [states[code]["lons"] for code in states if code not in EXCLUDED]
state_ys = [states[code]["lats"] for code in states if code not in EXCLUDED]

network_graph.node_renderer.data_source.data['colors'] = color_map
network_graph.node_renderer.data_source.data['retcount'] = retcount
network_graph.node_renderer.data_source.data['sizes'] = srccnt
network_graph.edge_renderer.data_source.data['colors1'] = edge_colors
network_graph.edge_renderer.data_source.data['sizes1'] = total
network_graph.edge_renderer.data_source.data['sizes2'] = total1
network_graph.edge_renderer.data_source.data['totcount'] = totcount

network_graph.node_renderer.glyph = Circle(size='sizes', fill_color='colors')
network_graph.node_renderer.hover_glyph = Circle(size='sizes', fill_color='colors')
network_graph.node_renderer.selection_glyph = Circle(size='sizes', fill_color='colors')
network_graph.edge_renderer.glyph = MultiLine(line_color='colors1', line_alpha=0.8, line_width='sizes1')
network_graph.edge_renderer.selection_glyph = MultiLine(line_color='colors1', line_width='sizes2')
network_graph.edge_renderer.hover_glyph = MultiLine(line_color='colors1', line_width='sizes2')
network_graph.selection_policy = NodesAndLinkedEdges()
network_graph.inspection_policy = NodesAndLinkedEdges()
plot.renderers.append(network_graph)
plot.patches(state_xs, state_ys, fill_alpha=0.0,
          line_color="#884444", line_width=2, line_alpha=0.3)
plot.circle(30,-125,legend_label="Travel State", color="green")
plot.line(26,-125,legend_label=">1000 miles", line_color="orange", line_width=5)
plot.line(24,-125,legend_label="<1000 miles", line_color="lightseagreen", line_width=5)
plot.legend.location = 'bottom_left'
hover = HoverTool(tooltips=[("State Name", "@index"),("Travellers retained", "@retcount")], 
                  renderers=[network_graph])
hover_no_tooltips = HoverTool(
    tooltips=None, renderers=[network_graph])
plot.add_tools(hover, hover_no_tooltips, TapTool(), BoxSelectTool())
show(plot)
