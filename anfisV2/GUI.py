import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
import threading
import testV2
import myANFIS_V2

# Color scheme
BG_COLOR = "#1e1e2e"
FG_COLOR = "#ffffff"
ACCENT_COLOR = "#00e5ff"
BUTTON_COLOR = "#00ff85"
ENTRY_BG = "#32324e"

root = tk.Tk()
root.title("AI ANFIS Model Trainer")
root.configure(bg=BG_COLOR)
root.geometry("750x650")

# Font
FONT = ("Segoe UI", 11)
FONT_BOLD = ("Segoe UI", 11, "bold")

# Functions
def load_file():
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if filepath:
        entry_path.delete(0, tk.END)
        entry_path.insert(0, filepath)
        log_message(f"File loaded: {filepath}")

def start_train_thread():
    threading.Thread(target=start_train, daemon=True).start()

def start_train():
    try:
        progress.start(10)
        log_message("Starting training...")

        epoch_n = int(entry_epoch.get())
        mf = int(entry_mf.get())
        step_size = float(entry_step_size.get())
        decrease_rate = float(entry_decrease.get())
        increase_rate = float(entry_increase.get())
        filepath = entry_path.get()

        if not filepath:
            log_message("Error: No CSV file selected.")
            progress.stop()
            return

        testV2.run_test(filepath, epoch_n, mf, step_size, decrease_rate, increase_rate, log)

        log_message("Training completed successfully.")
        log_message("By AP_Lab")

    except ValueError as e:
        log_message(f"Value Error: {str(e)}")
    except Exception as e:
        log_message(f"Unexpected error: {str(e)}")
    finally:
        progress.stop()

def log_message(message):
    log.config(state=tk.NORMAL)
    log.insert(tk.END, message + "\n")
    log.see(tk.END)
    log.config(state=tk.DISABLED)

# GUI Components
tk.Label(root, text="AI ANFIS Model Trainer", bg=BG_COLOR, fg=ACCENT_COLOR, font=("Segoe UI", 16, "bold")).pack(pady=10)

frame = tk.Frame(root, bg=BG_COLOR)
frame.pack(pady=10)

tk.Label(frame, text="CSV Path:", bg=BG_COLOR, fg=FG_COLOR, font=FONT_BOLD).grid(row=0, column=0, sticky='e', padx=5, pady=5)
entry_path = tk.Entry(frame, width=40, bg=ENTRY_BG, fg=FG_COLOR, font=FONT)
entry_path.grid(row=0, column=1, padx=5)
tk.Button(frame, text="Load CSV", command=load_file, bg=BUTTON_COLOR, fg=BG_COLOR, font=FONT_BOLD).grid(row=0, column=2, padx=5)

labels = ["Epochs:", "MF:", "Step Size:", "Decrease Rate:", "Increase Rate:"]
default_values = ["20", "3", "0.1", "0.9", "0.1"]
entries = []

for i, label in enumerate(labels):
    tk.Label(frame, text=label, bg=BG_COLOR, fg=FG_COLOR, font=FONT_BOLD).grid(row=i+1, column=0, sticky='e', padx=5, pady=5)
    entry = tk.Entry(frame, bg=ENTRY_BG, fg=FG_COLOR, font=FONT)
    entry.grid(row=i+1, column=1, padx=5, pady=5)
    entry.insert(0, default_values[i])
    entries.append(entry)

entry_epoch, entry_mf, entry_step_size, entry_decrease, entry_increase = entries

tk.Button(root, text="Start Training", command=start_train_thread, bg=BUTTON_COLOR, fg=BG_COLOR, font=FONT_BOLD, width=20).pack(pady=10)

progress = ttk.Progressbar(root, mode="indeterminate")
progress.pack(pady=5, fill='x', padx=20)

log = ScrolledText(root, width=95, height=30, state=tk.DISABLED, bg=ENTRY_BG, fg=FG_COLOR, font=("Consolas", 10))
log.pack(pady=10)

root.mainloop()
