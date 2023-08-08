import numpy as np
import regex as re
import pandas as pd
import yaml
import sys

sys.path.append('..')
from helper_funcs.helper_funcs import util as hf

cfg = hf.DotDict(yaml.safe_load(open('config.yml')))

# helper function that adds columns relevant to a date (e.g. is_month_end)
# source: https://docs.fast.ai/tabular.core.html#add_datepart
def make_date(df, date_field):
    "Make sure `df[date_field]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)


def add_datepart(df, field_name, prefix=None, drop=True, time=False):
    "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
    make_date(df, field_name)
    field = df[field_name]
    if prefix is None:
        prefix = re.sub('[Dd]ate$', '', field_name)
    #prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    # Pandas removed `dt.week` in v1.1.10
    week = field.dt.isocalendar().week.astype(field.dt.day.dtype) if hasattr(field.dt, 'isocalendar') else field.dt.week
    for n in attr: df[prefix + n] = getattr(field.dt, n.lower()) if n != 'Week' else week
    mask = ~field.isna()
    df[prefix + 'Elapsed'] = np.where(mask,field.values.astype(np.int64) // 10 ** 9,np.nan)
    if drop: df.drop(field_name, axis=1, inplace=True)
    return df


'''
    add lags up to N number of days to use as features
    the lag columns are labelled as 'adj_close_lag_1', ... etc.
'''
def add_lags(df, N, lag_cols):
    # use lags up to N number of days to use as features
    df_w_lags = df.copy()
    df_w_lags.loc[:, 'order_day'] = [x for x in list(range(len(df)))] # add a column 'order_day' to indicate the order of the rows by date
    merging_keys = ['order_day']
    shift_range = [x+1 for x in range(N)]
    for shift in shift_range:
        train_shift = df_w_lags[merging_keys + lag_cols].copy()

        # e.g. order day of 0 becomes 1 for shift = 1
        # so when this is merged with order_day of 1 in df_w_lags, this will represent lag of 1.
        train_shift['order_day'] = train_shift['order_day'] + shift

        foo = lambda x: '{}_lag_{}'.format(x, shift) if x in lag_cols else x
        train_shift = train_shift.rename(columns=foo)

        df_w_lags = pd.merge(df_w_lags, train_shift, on=merging_keys, how='left')
    
    return df_w_lags


def chart_output(
    fig,
    figpre,
    lognamestr, 
    loggername='__main__',
    chart_height=cfg.chart_output.HEIGHT,
    chart_width=cfg.chart_output.WIDTH,
    chart_scale=cfg.chart_output.SCALE,
    write_html = cfg.chart_output.WRITE,
    write_img = cfg.chart_output.IMAGE,
    watermark=True,
    save_short=True
):
    """
    write out or show charts according to config settings
    
    fig: (obj) the plotly figure object
    figpre: (str) prefix of figure to uniquely identify in the run
    lognamestr: (str) the logname determined in the calling function, e.g. dirname_scriptname_timestamp
    loggername: (str) the name property of the logger object
    chart_height: (int) output pixel height of images
    chart_width: (int) output pixel width of images
    chart_scale: (float) scale factor applied to images. Larger increases text size
    watermark: (bool) whether to include the chart name as an annotation in the chart
    save_short: (bool) whether to include the full logname string when saving the file to logs
    """
    
    loc_logger = logging.getLogger(loggername)
    figname = f"{figpre}_{lognamestr}"

    if cfg.chart_output.SHOW: fig.show() 
    
    if watermark:  # only for file output
        fig.add_annotation(
            text=figname,
            xref="paper",
            yref="paper",
            x=1,
            y=0,
            showarrow=False,
            xanchor='right',
            yanchor='bottom',
            textangle=-90,
            font={'size':6, 'color':'lightgrey'},
        )
    if save_short: figname = f"{figpre}"
    if write_html: fig.write_html(f"{cfg.out.FIGS}/{lognamestr}/{figname}.html")
    if write_img:
        fig.write_image(
            f"{cfg.out.IMGS}/{lognamestr}/{figname}.png",
            width=chart_width/chart_scale,
            height=chart_height/chart_scale,
            scale=chart_scale
        )
    loc_logger.info(f'{figname} produced')
