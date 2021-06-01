import warnings
warnings.filterwarnings('ignore')

from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
from os.path import join
import pandas as pd
import numpy as np
import pathlib
import json

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import dash
import plotly.graph_objects as go

import torch
import plotly.figure_factory as ff

from scipy.interpolate import interp1d

import sys
sys.setrecursionlimit(1500)


################################### helper functions ###################################

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()

    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[torch.isnan(loss)] = 0

    return loss.mean()


def predictions_to_df(horizon, timestamp_value):
    odf=pd.DataFrame(columns=["pred_speed", "pred_error"])

    try: # timestamp gets converted to string when callback
        timestamp_value = datetime.strptime(timestamp_value, '%Y-%m-%dT%H:%M:%S')
    except:
        pass

    tidx = np.where(np.array(data_pred["time"]) == timestamp_value)[0][0]
    
    for sid in data_pred[horizon].keys():
        isid=int(sid)
        #print(sid)
        try:
            odf.loc[isid,"pred_speed"] = float(data_pred[horizon][sid].loc[tidx,"pred"])
            odf.loc[isid,"true_speed"] = float(data_pred[horizon][sid].loc[tidx,"truth"])
            odf.loc[isid,"pred_error"] = float(data_pred[horizon][sid].loc[tidx,"pred"] - data_pred[horizon][sid].loc[tidx,"truth"])
        except AttributeError:
            pass
    
    return odf


def plot_time_series_station(stid, horizon, start_date, end_date):
    tdf=get_time_series(stid, horizon, start_date, end_date)

    if type(tdf) == type(None):
        print("got None time series for %d" % stid)
        return None
    
    fig3 = go.Figure()
    mae_error = masked_mae_loss(torch.tensor(tdf["pred"].values), torch.tensor(tdf["truth"].values))
    fig3.add_trace(go.Scatter(x=tdf["time"], y=tdf["truth"], mode='lines', name='True'))
    fig3.add_trace(go.Scatter(x=tdf["time"], y=tdf["pred"], mode='lines', name='Predicted (MAE: {:.2f}mph)'.format(mae_error)))
    # fig3.write_image("test_output.jpg")
    fig3.update_layout(
        title="<b>Time Series for Sensor {} with {}mins Forecasting Horizon </b>".format(stid, horizon*5),
        # xaxis_title="Time",
        yaxis_title='<b>Average Speed (MPH)</b>',
        height=600, template="plotly_dark", 
        plot_bgcolor= 'rgba(0, 0, 0, 0)', 
        paper_bgcolor= 'rgba(0, 0, 0, 0)',
        font=dict(size=14, color='white'),
        margin={"r":50,"t":50,"l":50,"b":20})

    fig3.update_xaxes(rangeslider_visible=True, showgrid=False)
    fig3.update_yaxes(showgrid=False)

    min_date = tdf.time.min()
    max_date = tdf.time.max()

    return fig3, min_date, max_date


#function to move fwys in separate directions apart for visualization
def spread_lanes(sdf, fwdir, latspread=0.0005, lonspread=0.0005):
    direc=fwdir[-1]
    
    # this assumes right-hand traffic
    if direc == 'W':
         sdf["latitude"] += latspread
    elif direc == "E":
        sdf["latitude"] -= latspread
    elif direc == "S":
        sdf["longitude"] -= lonspread
    elif direc == "N":
        sdf["longitude"] += lonspread
    
    return sdf


hrdfs=[]
def interpolate_onto_hr_roads(indf, cols=["avg_speed"], dropna = False, do_spread_lanes=True):

    for fwdir in indf["fwdir"].unique():
        print("\rInterpolating average speed along %s " % fwdir, end="")
        sel=indf.query('fwdir == "%s"' % fwdir).sort_values(by="abs_pm")
        #selhr=pd.read_sql("select * from usdot u where u.fwdir = '%s' order by u.abs_pm" % fwdir, engine)
        selhr=hfroads_data.query("fwdir == '%s'" % fwdir).sort_values(by="abs_pm")
        
        if do_spread_lanes:
            selhr=spread_lanes(selhr, fwdir)

        if len(sel) > 1:
            #fillvalmin=sel.loc[minidx,"abs_pm"]
            #fillvalmax=sel.loc[maxidx,"abs_pm"]
            for col in cols:
                interpmod = interp1d(sel["abs_pm"].values, sel[col].values, bounds_error=False)
                selhr[col] = interpmod(selhr["abs_pm"].values)
                if dropna:
                    selhr.dropna(axis="index", how="any", subset=[col], inplace=True)
            hrdfs.append(selhr)
    print(" done.")

    selhr=pd.concat(hrdfs)
    selhr.reset_index(drop=True, inplace=True)
    
    return selhr


def speed2cat(speed):
    if np.isnan(speed):
        return 'no data'
    elif speed > 65:
        return 'fast'
    elif speed > 60:
        return 'moderate'
    elif speed > 50:
        return 'slow'
    else: 
        return 'very slow'


def err2cat(err):
    if np.isnan(err):
        return 'no data'
    elif err > 5:
        return 'overpredict > 5 mph'
    elif err > 1:
        return 'overpredict 1 - 5 mph'
    elif err > -1: 
        return 'within 1 mph'
    elif err > -5:
        return 'underpredict 1 - 5 mph'
    else:
        return 'underpredict > 5 mph'


 # %load road_group_split.py
def road_group_split(indf, splitcat="pred_speed_cat" ):
                     #keep_cols=["latitude", "longitude", "abs_pm", "pred_speed","fwdir"]):
    
    odict={}
    
    keep_cols=list(indf.columns)
    keep_cols.remove(splitcat)
    
    for col in keep_cols + [splitcat]:
        odict[col] = []
    #lats=[]
    #lons=[]
    #abs_pm=[]
    #cat=[]
    odict["linegroup"] = []
    
    p=0 #number holding line group
    i0=indf.index[0]
    
    for col in keep_cols + [splitcat]:
        odict[col].append(indf.loc[i0,col])
    
    #lats.append(indf.loc[i0, "latitude"])
    #lons.append(indf.loc[i0, "longitude"])
    #abs_pm.append(indf.loc[i0, "abs_pm"])
    #cat.append(indf.loc[i0, splitcat])
    #group.append(p)
    odict["linegroup"].append(p)
    
    for n in indf.index[1:]:
        if indf.loc[n,"fwdir"] != indf.loc[i0,"fwdir"]:
            p+=1
            
        if indf.loc[n,splitcat] != indf.loc[i0,splitcat] and indf.loc[n,"fwdir"] == indf.loc[i0,"fwdir"]:
            for col in keep_cols:
                odict[col].append(indf.loc[n,col])
            odict[splitcat].append(indf.loc[i0,splitcat])
            #end previous group with new point and speed cat from previous point
            #lats.append(indf.loc[n,"latitude"])
            #lons.append(indf.loc[n,"longitude"])
            #lons.append(indf.loc[n,"abs_pm"])
            #cat.append(indf.loc[i0,splitcat])
            #group.append(p)
            odict["linegroup"].append(p)
            p+=1
        #start new group
        for col in keep_cols + [splitcat]:
            odict[col].append(indf.loc[n,col])
        #lats.append(indf.loc[n,"latitude"])
        #lons.append(indf.loc[n,"longitude"])
        #cat.append(indf.loc[n,splitcat])
        #group.append(p)        
        odict["linegroup"].append(p)
        i0=n
    
    #odf=pd.DataFrame({"latitude": lats, "longitude": lons, splitcat: cat, "linegroup": group})
    odf=pd.DataFrame(odict)
    return odf     


def get_time_series(stid, horizon, start_date, end_date):
    tdf=pd.DataFrame(columns=["pred", "truth", "time"])
    tdf["time"]=data_pred["time"]
    
    if str(stid) in data_pred[horizon].keys(): 
        tdf["pred"] = data_pred[horizon][str(stid)]["pred"]
        tdf["truth"] = data_pred[horizon][str(stid)]["truth"]

        #filter for one day - hardcoding for now
        idx=(tdf["time"] >= start_date) & (tdf["time"] <= end_date)
    
        return tdf[idx]
    
    else:
        print('station %d not in predictions' % stid)
        return None


def get_timestamp_list(start_date, end_date, t_delta):
    t = start_date
    timestamp_list=[]
    delta = timedelta(minutes=t_delta)
    
    while t <= end_date:
        timestamp_list.append(t)
        t += delta

    return timestamp_list


def find_selhr(snap_var):
    selhr=interpolate_onto_hr_roads(snap_var, cols=["pred_speed", "true_speed", "pred_error"])

    selhr["pred_speed_cat"]=selhr["pred_speed"].apply(speed2cat)
    selhr["true_speed_cat"]=selhr["true_speed"].apply(speed2cat)

    selhr["err_cat"]=selhr["pred_error"].apply(err2cat)

    return selhr

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


################################### Initialize app and variables ###################################

app = dash.Dash(external_stylesheets=[dbc.themes.SLATE, 'https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

# load data
#APP_PATH = str(pathlib.Path(__file__).parent.resolve())

# EDA
bay_data = pd.read_csv('bay_data.csv')#pd.read_csv(join(APP_PATH, 'bay_data.csv'))
sd_data = pd.read_csv( 'sd_data.csv')#pd.read_csv(join(APP_PATH, 'sd_data.csv')) 
traffic_bay_data = pd.read_csv('traffic_bay.csv') #pd.read_csv(join(APP_PATH, 'traffic.csv')) 
traffic_sd_data = pd.read_csv('traffic_sd.csv') #pd.read_csv(join(APP_PATH, 'traffic.csv')) 
traffic_covid_sd_data = pd.read_csv('traffic_covid_sd.csv') #pd.read_csv(join(APP_PATH, 'traffic_covid.csv')) 
traffic_covid_sd_data['norm_new_cases'] = NormalizeData(traffic_covid_sd_data.new_cases)
traffic_covid_sd_data['norm_min_avg_speed'] = NormalizeData(traffic_covid_sd_data.avg_avg_speed)
traffic_covid_bay_data = pd.read_csv('traffic_covid_bay.csv') #pd.read_csv(join(APP_PATH, 'traffic_covid.csv')) 
traffic_covid_bay_data['norm_new_cases'] = NormalizeData(traffic_covid_bay_data.new_cases)
traffic_covid_bay_data['norm_min_avg_speed'] = NormalizeData(traffic_covid_bay_data.avg_avg_speed)

# traffic predictions
data_pred = pd.read_pickle('traffic_pred_horizon12.pkl') #pd.read_pickle(join(APP_PATH, 'traffic_pred_horizon12.pkl')) 
sensors_data = pd.read_csv('sensors_data.csv').set_index('sid') #pd.read_csv(join(APP_PATH, 'sensors_data.csv')).set_index('sid')
hfroads_data = pd.read_csv('hfroads_data.csv').set_index('usdid') #pd.read_csv(join(APP_PATH, 'hfroads_data.csv')).set_index('usdid')
    
# misc variables
cdm={"fast": "green",
     "moderate": "orange",
     "slow": "red",
     "very slow": "brown",
     "no data": "gray"}

cdme={"within 1 mph": "green",
     "underpredict 1 - 5 mph": "orange",
     "underpredict > 5 mph": "red",
     "overpredict 1 - 5 mph": "lightgreen",
     "overpredict > 5 mph": "darkgreen",
     "no data": "gray"}

end_date = max(data_pred['time'])
start_date = end_date - relativedelta(days=1)

timestamp_list = get_timestamp_list(start_date, end_date, 5)

initial_horizon = 12
initial_timestamp = timestamp_list[-1]

odf=predictions_to_df(initial_horizon, initial_timestamp)

# print(type(sensors_data))
snap=pd.merge(sensors_data, odf, left_index=True, right_index=True)

tstart_date = '2020-01-01'
tend_date = '2020-12-31'

#snap=sensors_data.merge(odf)


################################### layout functions ###################################

buttons_style = {'font-weight': 'bold', 'color': 'white', 'font-size': '15px'}

# main dashboard description and EDA
def main_title_buttons():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.Div([
                        html.Div([
                            html.Br(),
                            html.H1(children="Robust Traffic Predictions in Uncertain Times Dashboard", style={'font-weight': 'bold'}),
                            # html.Br(),
                            html.P(                                 
                                children="This Dashboard is telling a story via Exploratory Data Analysis for both the traffic and COVID-19 datasets  \
                                and provides time series spatio-temporal model traffic forecasting prediction visualizations."),
                            html.Br(),
                            html.P(
                                children="The Exploratory Data Analysis include following views: "),
                            html.P(children="a. Average Speed by Time of Day (averaged over 5 min window).", style={'margin-left': '2%'}),
                            html.P(children="b. View to analyze the traffic speed with respect to new COVID-19 cases by days in 2020.", style={'margin-left': '2%'}),
                            html.P(children="c. Pearson correlation matrix comparing COVID-19 and Caltrans traffic dataset.", style={'margin-left': '2%'}),    
                            html.Br(),
                            html.P(children='Information presented can be narrowed down to a specific date range and/or filtered for one of the two counties, San Diego and Bay.'),
                            html.Br(),
                        ], style={'width': '100%', 'color': 'white', 'font-size': '20px'}),
                        html.Div([
                            html.P('Select Date Range:', style=buttons_style),
                            dcc.DatePickerRange(
                                id='date-range',
                                start_date_placeholder_text="Start Period",
                                end_date_placeholder_text="End Period",
                                calendar_orientation='vertical',
                                #min_date_allowed=min(data_pred['time']),
                                #max_date_allowed=max(data_pred['time']),
                                start_date=tstart_date,
                                end_date=tend_date,
                                day_size=45,
                            )
                        ], style={'width': '21%', 'margin-left': '2%', 'color': 'white'}),
                        html.Div([
                            html.P('Select County:', style=buttons_style),
                            dcc.Dropdown(
                                id="county1",
                                options=[
                                    {'label': 'San Diego County', 'value': 73},
                                    {'label': 'Bay County', 'value': 85}
                                ],
                                multi=False,
                                clearable=False,
                                value=73
                            )  
                        ], style={'width': '15%', 'margin-left': '2%'}),
                    ], className="row"),
                ], style={'textAlign': 'left', 'margin-left': '2%', 'margin-right': '2%'}),
                html.Div([html.Br()]),
            ])
        ),
    ])


# prediction analysis buttons
def timeseries_title_buttons():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.Div([
                        html.Div([
                            # html.H1(children="Environmental Radiation Monitoring in the US"),
                            html.Br(),
                            html.P(
                                children="Use the map below and time series to explore DCRNN traffic spatio-temporal predictions."),
                            html.Br(),
                            html.P(
                                children="Traffic variables for true, predicted, and error as well as the snapshot timestamp options will \
                                update the map on the left, while the date range, horizon, and station options will update both figures. \
                                Main Line (ML) legend, blue dots, \
                                within the map are clickable and can be used to select a station and update the timeseries on the right, using the \
                                dropdown menu option will accomplish the same task."),
                            # html.P('Information presented in this map is grouped by the selecion on the "Group By" option. The map and charts will \
                            #         update accordingly.'),
                            html.Br(),
                        ], style={'width': '100%', 'color': 'white', 'font-size': '20px'}),
                        html.Div([
                            html.P('Select Variable:', style=buttons_style),
                            dcc.RadioItems(
                                id='plottype',
                                options=[{"label":"True traffic","value": "true_speed_cat"},
                                         {"label":"Predicted traffic", "value": "pred_speed_cat"},
                                         {"label":"Prediction error", "value": "err_cat"}],
                                value = 'pred_speed_cat',
                                labelStyle={'display': 'block', 'color': 'white'}
                                )
                        ], style={'width': '10%', 'margin-left': '2%'}), 
                        html.Div([
                            html.P('Select Map Snapshot Timestamp:', style=buttons_style),
                            dcc.Dropdown(
                                id='snapshot-timestamp',
                                options=[{"label": str(i), "value": i} for i in timestamp_list],
                                value = initial_timestamp,
                                # placeholder='Select'
                            )
                        ], style={'width': '15%', 'margin-left': '1%', 'margin-right': '10%'}),   
                        html.Div([
                            html.P('Select Date Range:', style=buttons_style),
                            dcc.DatePickerRange(
                                id='pred-date-range',
                                start_date_placeholder_text="Start Period",
                                end_date_placeholder_text="End Period",
                                calendar_orientation='vertical',
                                min_date_allowed=min(data_pred['time']),
                                max_date_allowed=max(data_pred['time']),
                                start_date=start_date,
                                end_date=end_date,
                                day_size=45,
                            )
                        ], style={'width': '20%', 'margin-left': '1%'}),                        
                        html.Div([
                            html.P('Select Horizon (5 mins/pts):', style=buttons_style),
                            dcc.Dropdown(
                                id='horizons-select',
                                options=[{'label':i+1, 'value':i+1} for i in range(12)],
                                value=initial_horizon,
                                multi=False
                            ) 
                        ], style={'width': '15%', 'margin-left': '1%'}),
                        html.Div([
                            html.P('Select Station:', style=buttons_style),
                            dcc.Dropdown(
                                id='station-dropdown',
                                options=[{"label": sid, "value": sid} for sid in list(snap.index)],
                                value =snap.index[0]
                            )
                        ], style={'width': '15%', 'margin-left': '2%'}),

                    ], className="row"),
                ], style={'textAlign': 'left', 'margin-left': '2%', 'margin-right': '2%'}),
                html.Div([html.Br()]), 
            ])
        ),
    ])


# functions for plots
def traffic_animated_plot():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    id="traffic-animated-plot"
                ) 
            ])
        ),  
    ])


def traffic_covid():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    id="traffic-covid-plot"
                ) 
            ])
        ),  
    ])


def correlations_plot():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    id="correlations-plot"
                )
            ]), 
        ),  
    ])


def roadmap_plot():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    id="roadmap-plot"
                ) 
            ])
        ),  
    ])


def station_timeseries():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    id="station-timeseries-plot"
                ) 
            ])
        ),  
    ])

################################### layout ###################################

# App layout
app.layout = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    main_title_buttons()
                ], width=12),
            ], align='center'), 
            html.Br(),
            dbc.Row([
                dbc.Col([
                    traffic_animated_plot() 
                ], width=7),
                dbc.Col([
                    correlations_plot()
                ], width=5),
            ], align='center'),  
            html.Br(), 
            dbc.Row([
                dbc.Col([
                    traffic_covid() 
                ], width=12),
            ], align='center'), 
            html.Br(),
            dbc.Row([
                dbc.Col([
                    timeseries_title_buttons(),
                ], width=12)
            ], align='center'),  
            html.Br(),
            dbc.Row([
                dbc.Col([
                    roadmap_plot() 
                ], width=6),
                dbc.Col([
                    station_timeseries()
                ], width=6),
            ], align='center'),   
        ]), color = 'dark'
    ),
    # Hidden div inside the app that stores the intermediate value
    html.Div(id='map-snapshot', style={'display': 'none'}),
    html.Pre(id='click-data')
])


@app.callback(
    Output("traffic-animated-plot", "figure"),
    [
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date'),
        Input('county1', 'value')
    ]
)
def plot_traffic_animated(start_date, end_date, value):
    start_date1 = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date1 = datetime.strptime(end_date, '%Y-%m-%d').date()
    if value == 73: 
        traffic_data_subset = traffic_sd_data[(pd.to_datetime(traffic_sd_data.tmonth, format='%Y%m', errors='coerce').dt.date >= start_date1) & (pd.to_datetime(traffic_sd_data.tmonth, format='%Y%m', errors='coerce').dt.date <= end_date1)]
        title1 = "<b>Average Speed by Time of Day in 2020 for San Diego County</b>"
    else:
        traffic_data_subset = traffic_bay_data[(pd.to_datetime(traffic_bay_data.tmonth, format='%Y%m', errors='coerce').dt.date >= start_date1) & (pd.to_datetime(traffic_bay_data.tmonth, format='%Y%m', errors='coerce').dt.date <= end_date1)]
        title1 = "<b>Average Speed by Time of Day in 2020 for Bay County</b>"
    fig = px.line(traffic_data_subset,  
                 x ='ttime',  
                 y =['aspeed'], 
                 color ='tmonth',
                 line_group = 'tmonth',
                 animation_group ='tmonth', 
                 animation_frame = 'tmonth',
                 labels = {'ttime':'<b>Time of the Day</b>',
                           'value':'<b>Average Speed</b>',
                           'tmonth':'<b>Month</b>'},
                 #hover_name ='ttime',  
                 log_y = True,
                 # width = 900,
                 height = 600,
                 range_y = [50,70],
                 #y_title = "Avg Speed by month in 5 min interval',
                 title = title1)
    
    fig.update_layout(template="plotly_dark", 
                      plot_bgcolor= 'rgba(0, 0, 0, 0)', 
                      paper_bgcolor= 'rgba(0, 0, 0, 0)',
                      font=dict(size=11, color='white'),
                      # height=600,
                      margin={"r":50,"t":30,"l":50,"b":0})
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


@app.callback(
    Output("traffic-covid-plot", "figure"),
    [
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date'),
        Input('county1', 'value')
    ]
)
def plot_traffic_covid(start_date, end_date, value):
    start_date1 = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date1 = datetime.strptime(end_date, '%Y-%m-%d').date()
    if value == 73:
        traffic_covid_subset = traffic_covid_sd_data[(pd.to_datetime(traffic_covid_sd_data.record_date, format='%Y-%m-%d', errors='coerce').dt.date >= start_date1) & (pd.to_datetime(traffic_covid_sd_data.record_date, format='%Y-%m-%d', errors='coerce').dt.date <= end_date1)]
        title1 = "<b>Normalized Average Speed and Covid Cases by Day for San Diego County</b>"
    else:
        traffic_covid_subset = traffic_covid_bay_data[(pd.to_datetime(traffic_covid_bay_data.record_date, format='%Y-%m-%d', errors='coerce').dt.date >= start_date1) & (pd.to_datetime(traffic_covid_bay_data.record_date, format='%Y-%m-%d', errors='coerce').dt.date <= end_date1)]
        title1 = "<b>Normalized Average Speed and Covid Cases by Day for Bay County</b>"
    fig = px.line(traffic_covid_subset,
                 x ='record_date', 
                 y =['norm_min_avg_speed','norm_new_cases'], 
                 #color = 'rmonth',
                 #animation_group ='rmonth', 
                 #animation_frame = 'rmonth',
                 #hover_name ='rmonth', 
                 #log_y = True,
                 labels = {'record_date':'<b>Record Date</b>',
                           'value':'<b>Normalized Value</b>',
                           'variable':'<b>Measure</b>'},
                 # width = 1100,
                 height = 600,
                 range_y = [0.01, 1],
                 title = title1)

    fig.update_layout(
        margin=dict(t=100, b=70, l=250, r=50),
        updatemenus=[
            dict(
                type="buttons",
                x=-0.07,
                y=0.7,
                showactive=False,
                buttons=list(
                    [
                        dict(
                            label="Both",
                            method="update",
                            args=[{"y": [traffic_covid_subset["norm_min_avg_speed"], traffic_covid_subset["norm_new_cases"]]}],
                        ),
                        dict(
                            label="Normalized Minimum Average Speed",
                            method="update",
                            args=[{"y": [traffic_covid_subset["norm_min_avg_speed"]]}],
                        ),
                        dict(
                            label="Normalized New Cases",
                            method="update",
                            args=[{"y": [traffic_covid_subset["norm_new_cases"]]}],
                        ),
                        
                    ]),
            )])
    
    fig.update_layout(template="plotly_dark", plot_bgcolor= 'rgba(0, 0, 0, 0)', paper_bgcolor= 'rgba(0, 0, 0, 0)',
                        transition_duration=2000, font=dict(size=14, color='white'), margin={"r":50,"t":50,"l":50,"b":20})
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


@app.callback(
    Output("correlations-plot", "figure"),
    [
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date'),
        Input('county1', 'value')
    ]
)
def plot_correlations(start_date, end_date, value):
    start_date1 = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date1 = datetime.strptime(end_date, '%Y-%m-%d').date()
    if value == 73:         
        corr_subset = sd_data[(pd.to_datetime(sd_data.record_date, format='%Y-%m-%d', errors='coerce').dt.date >= start_date1) & (pd.to_datetime(sd_data.record_date, format='%Y-%m-%d', errors='coerce').dt.date <= end_date1)]
        title1='<b>San Diego County Traffic and Covid Pearson Correlation</b>'
    else:
        corr_subset = bay_data[(pd.to_datetime(bay_data.record_date, format='%Y-%m-%d', errors='coerce').dt.date >= start_date1) & (pd.to_datetime(bay_data.record_date, format='%Y-%m-%d', errors='coerce').dt.date <= end_date1)]
        title1='<b>Bay County Traffic and Covid Pearson Correlation</b>'
    
    correlations = corr_subset.corr(method='pearson')

    correlations.new_cases = correlations.new_cases.apply(lambda x: round(x, 6))
    correlations.new_deaths = correlations.new_deaths.apply(lambda x: round(x, 6))
    correlations.tot_total_flow = correlations.tot_total_flow.apply(lambda x: round(x, 6))
    correlations.avg_avg_speed = correlations.avg_avg_speed.apply(lambda x: round(x, 6))
    
    z = correlations.values.tolist()
    x = correlations.columns.tolist()
    y = correlations.index.tolist()
    
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z, colorscale='viridis')

    # for add annotation into Heatmap
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 12

    fig.update_layout(title_text=title1, 
                      height=600, template="plotly_dark", 
                      plot_bgcolor= 'rgba(0, 0, 0, 0)', 
                      paper_bgcolor= 'rgba(0, 0, 0, 0)',
                      font=dict(size=14, color='white'),
                      margin={"r":50,"t":50,"l":50,"b":20})
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    
    return fig


@app.callback(
Output('roadmap-plot','figure'),
[Input('plottype', 'value'),
Input('map-snapshot', 'children'),
Input('snapshot-timestamp', 'value'),
Input('horizons-select','value')])
def plot_roads(plotcat, snap_var, timestamp_value, horizon):
    # plotly.io.templates.default='plotly'
    #del fig
    
    snap_var = pd.read_json(snap_var, orient='split')

    selhr = find_selhr(snap_var)

    selhrg=road_group_split(selhr, splitcat=plotcat) 
                        #keep_cols=keepcols)

    fig = px.line_mapbox(
    #fig=px.scatter_mapbox
        selhrg, lat="latitude", lon="longitude", color=plotcat,
                            #mapbox_style="carto-darkmatter",
                            mapbox_style="carto-positron",
                            #color_continuous_scale="aggrnyl" ,
                            line_group = "linegroup",
                            hover_data=list(selhrg.columns),
                            center = {"lat": 37.34, "lon": -121.93},
                            color_discrete_map= cdme if plotcat == "err_cat" else cdm,
                            zoom=10
                            # height=600, width=600
                            )

    for dat in fig.data:
        dat["line"]["width"] = 5

    sensors_data["size"] = 3
    fig2 = px.scatter_mapbox(sensors_data.reset_index(), lat="latitude", lon="longitude",
                            #hover_data=["stype","fwy","direc","abs_pm","pred_speed"],
                            hover_name="sid",
                            hover_data=list(sensors_data.reset_index().columns),
                            mapbox_style="stamen-terrain",
                            color="stype",
                            color_discrete_map={"ML": "blue"},
                            size="size",
                            opacity=0.35,
                            size_max=5
                            )
    #for dat in fig2.data:
    fig.add_trace(fig2.data[0])
    
    fig.update_layout(title="<b>PEMS Sensor Map Snapshot for {} with {}pts Horizon</b>".format(timestamp_value, horizon),
                      uirevision=True,
                      height=600, 
                      # width=1200,
                      autosize=True,
                      template="plotly_dark", 
                      plot_bgcolor= 'rgba(0, 0, 0, 0)', 
                      paper_bgcolor= 'rgba(0, 0, 0, 0)',
                      font=dict(size=14, color='white'),
                      margin={"r":40,"t":50,"l":10,"b":10})

    return  fig


@app.callback(
[Output('station-timeseries-plot','figure'),
Output('snapshot-timestamp', 'options')],
[Input('station-dropdown','value'),
Input('horizons-select','value'),
Input('pred-date-range','start_date'),
Input('pred-date-range','end_date')])
def plot_time_series_dropdown(stid, horizon, start_date, end_date):
    fig4, min_date, max_date=plot_time_series_station(stid, horizon, start_date, end_date)
    timestamp_options = [{"label": str(i), "value": i} for i in get_timestamp_list(min_date, max_date, 5)]

    return fig4, timestamp_options


@app.callback(
Output('station-dropdown','value'),
[Input('roadmap-plot','clickData'),
Input('map-snapshot', 'children')],
[State('station-dropdown','value')])

def adjust_station_selector(clickData, snap_var, station_value):
#def plot_time_series(stid):
    oldval=station_value
    
    if type(clickData) == type(None):
        return oldval
    
    if "hovertext" not in clickData["points"][0].keys():
        print("Click directly onto a sensor to plot time series.")
        return oldval

    stid=clickData["points"][0]["hovertext"]
    # print(stid)
    
    snap = pd.read_json(snap_var, orient='split') 

    if stid in snap.index:
        return stid
    
    else:
        print("site %d was not predicted")
        return oldval


@app.callback(
    Output('click-data', 'children'),
    [Input('roadmap-plot', 'clickData')])
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)

@app.callback(
    Output('map-snapshot', 'children'),
    [Input('horizons-select', 'value'),
    Input('snapshot-timestamp', 'value')])
def update_map_snapshot(horizon, timestamp_value):

    odf=predictions_to_df(horizon, timestamp_value)
   
    snap=pd.merge(sensors_data, odf, left_index=True, right_index=True)

    #convert from object to float
    for col in ["pred_speed", "pred_error"]:
        snap[col]=snap[col].astype(float)

    #create new column with fwy and direction as string
    snap["fwdir"] = snap.apply(lambda x: str(x["fwy"]) + str(x["direc"]), axis=1)
   
    return snap.to_json(date_format='iso', orient='split')



if __name__ == "__main__":

    # app.run_server(debug=True, use_reloader=False)

    app.run_server(debug=True)
