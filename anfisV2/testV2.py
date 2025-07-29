import tkinter as tk
import numpy as np
import dask.dataframe as dd
import myANFIS_V2 as anfis
from sklearn.preprocessing import MinMaxScaler

def run_test(filepath, epoch_n, mf, step_size, decrease_rate, increase_rate, log_widget):
    # อ่านไฟล์ CSV ด้วย Dask (จะไม่โหลดข้อมูลทั้งหมดในหน่วยความจำในครั้งเดียว)
    df = dd.read_csv(filepath)

    # คำนวณข้อมูลที่โหลดจาก Dask (compute จะทำให้ข้อมูลเป็น Pandas DataFrame)
    data = df.compute().values
    
    #data = np.genfromtxt(filepath, delimiter=',')
    
    # Divide data into input and output
    inputs = data[:, :-1]  # All columns except the last one are inputs
    output = data[:, -1:]  # The last column is the output
    ndata = data.shape[0]  # Data length


    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_input = scaler.fit_transform(inputs)



    # ANFIS train
    bestnet, y_myanfis, RMSE = anfis.myanfis(data, inputs, epoch_n, mf, step_size, decrease_rate, increase_rate )

    y_myanfis = anfis.evalmyanfis(bestnet, inputs)

    anfis_predictions = y_myanfis

    # For classification problem ( Round outputs to int)
    anfis_predictions = np.round(anfis_predictions).astype(int)

    # Calculate the RMSE
    rmse = anfis.calc_rmse(output,anfis_predictions)

    msg = f'Total RMSE error myanfis: {rmse:.2f}'
    print(msg)  # Print the message

    log_widget.config(state=tk.NORMAL)
    log_widget.insert(tk.END, msg)
    log_widget.config(state=tk.DISABLED)

    # anfis.plot_Nodes(bestnet)

    anfis.plot_mf(bestnet, data)

    anfis.plot_predictions(output,anfis_predictions)

    anfis.plot_r2(output,anfis_predictions)

    anfis.print_membership_functions(bestnet,log_widget)


