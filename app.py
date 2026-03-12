import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


def detect_frequency(index):
    inferred = pd.infer_freq(index)
    if inferred is not None:
        return inferred
    return "ME"


def summarize_trend(series):
    if len(series) < 12:
        return "The series is relatively short, so trend interpretation is limited."

    recent = series.iloc[-6:].mean()
    previous = series.iloc[-12:-6].mean()

    if recent > previous:
        return "Recent values suggest a mild upward trend."
    elif recent < previous:
        return "Recent values suggest a mild downward trend."
    return "Recent values suggest a relatively stable trend."


def get_columns(file_obj):
    if file_obj is None:
        return gr.update(choices=[]), gr.update(choices=[])

    df = pd.read_csv(file_obj.name, nrows=5)
    cols = list(df.columns)
    return gr.update(choices=cols), gr.update(choices=cols)


def run_forecast(file_obj, date_col, value_col, p, d, q, forecast_steps):
    if file_obj is None:
        raise gr.Error("Please upload a CSV file.")

    df = pd.read_csv(file_obj.name)

    if date_col not in df.columns or value_col not in df.columns:
        raise gr.Error("Please select valid columns from the uploaded CSV.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    df = df[[date_col, value_col]].dropna().sort_values(date_col)

    if len(df) < 20:
        raise gr.Error("Please upload a dataset with at least 20 valid rows.")

    ts = df.set_index(date_col)
    ts.index = pd.DatetimeIndex(ts.index)

    freq = detect_frequency(ts.index)

    test_size = max(12, int(len(ts) * 0.2))
    train = ts.iloc[:-test_size]
    test = ts.iloc[-test_size:]

    try:
        model = ARIMA(train[value_col], order=(int(p), int(d), int(q)))
        model_fit = model.fit()
    except Exception as e:
        raise gr.Error(f"Model fitting failed: {str(e)}")

    test_forecast = model_fit.forecast(steps=len(test))

    mae = mean_absolute_error(test[value_col], test_forecast)
    rmse = mean_squared_error(test[value_col], test_forecast) ** 0.5

    try:
        final_model = ARIMA(ts[value_col], order=(int(p), int(d), int(q))).fit()
        future_forecast = final_model.forecast(steps=int(forecast_steps))
    except Exception as e:
        raise gr.Error(f"Future forecasting failed: {str(e)}")

    last_date = ts.index[-1]
    future_index = pd.date_range(
        start=last_date,
        periods=int(forecast_steps) + 1,
        freq=freq
    )[1:]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(train.index, train[value_col], label="Train")
    ax.plot(test.index, test[value_col], label="Actual Test")
    ax.plot(test.index, test_forecast, label="Forecast on Test")
    ax.plot(future_index, future_forecast, label="Future Forecast")
    ax.set_title("Time Series Forecasting with ARIMA")
    ax.set_xlabel("Date")
    ax.set_ylabel(value_col)
    ax.legend()
    ax.grid(True)

    trend_summary = summarize_trend(ts[value_col])

    summary = (
        f"Rows used: {len(ts)}\n"
        f"Train rows: {len(train)} | Test rows: {len(test)}\n"
        f"ARIMA order: ({int(p)}, {int(d)}, {int(q)})\n"
        f"MAE: {mae:.3f}\n"
        f"RMSE: {rmse:.3f}\n\n"
        f"{trend_summary}\n"
        f"The next {int(forecast_steps)} forecasted periods are shown on the chart."
    )

    return fig, summary


with gr.Blocks() as demo:
    gr.Markdown("# Time Series Forecasting Studio")
    gr.Markdown(
        "Upload a CSV file, choose the date and value columns, set ARIMA parameters, and generate a forecast with evaluation metrics."
    )

    with gr.Row():
        file_input = gr.File(label="Upload CSV", file_types=[".csv"])
        date_col = gr.Dropdown(label="Date Column", choices=[])
        value_col = gr.Dropdown(label="Value Column", choices=[])

    file_input.change(
        fn=get_columns,
        inputs=file_input,
        outputs=[date_col, value_col]
    )

    with gr.Row():
        p = gr.Number(label="p", value=2, precision=0)
        d = gr.Number(label="d", value=1, precision=0)
        q = gr.Number(label="q", value=2, precision=0)
        forecast_steps = gr.Number(label="Forecast Steps", value=12, precision=0)

    run_button = gr.Button("Run Forecast")

    plot_output = gr.Plot(label="Forecast Plot")
    text_output = gr.Textbox(label="Model Summary", lines=10)

    run_button.click(
        fn=run_forecast,
        inputs=[file_input, date_col, value_col, p, d, q, forecast_steps],
        outputs=[plot_output, text_output]
    )

demo.launch()