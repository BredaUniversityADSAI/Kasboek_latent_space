import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pathlib import Path
import platform
import subprocess
import threading
import time
import os
from datetime import datetime
import csv
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sys
sys.path.append(str(Path(__file__).parent.parent))
from chat import LLMModel

class ArtInstallationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kasboek - Latent Space")
        self.root.geometry("800x600")
        
        # Create frames
        self.main_frame = ttk.Frame(root)
        self.manual_frame = ttk.Frame(root)
        self.contact_frame = ttk.Frame(root)
        self.logs_frame = ttk.Frame(root)
        self.docs_frame = ttk.Frame(root)
        self.dashboard_frame = ttk.Frame(root)
        self.config_frame = ttk.Frame(root)
        
        # Setup pages
        self.setup_main_page()
        self.setup_manual_page()
        self.setup_contact_page()
        self.setup_logs_page()
        self.setup_docs_page()
        self.setup_dashboard_page()
        self.setup_config_page()
        
        # Start automatic deletion thread
        self.start_automatic_deletion()
        
        # Show main page initially
        self.show_main_page()
    
    def setup_main_page(self):
        """Setup the main page"""
        title = ttk.Label(self.main_frame, text="Kasboek: Latent Space, Through the Eyes of the Algorithm", 
                         font=('Arial', 16, 'bold'))
        title.pack(pady=50)
        
        dashboard_button = ttk.Button(self.main_frame, text="Dashboard", 
                                     command=self.show_dashboard_page)
        dashboard_button.pack(pady=20)

        config_button = ttk.Button(self.main_frame, text="Configure Models", 
                           command=self.show_config_page)
        config_button.pack(pady=20)
        
        manual_button = ttk.Button(self.main_frame, text="User Manual", 
                                   command=self.show_manual_page)
        manual_button.pack(pady=20)
        
        contact_button = ttk.Button(self.main_frame, text="Contact Information", 
                                    command=self.show_contact_page)
        contact_button.pack(pady=20)
        
        logs_button = ttk.Button(self.main_frame, text="View Logs", 
                                command=self.show_logs_page)
        logs_button.pack(pady=20)
        
        docs_button = ttk.Button(self.main_frame, text="View Documents", 
                                command=self.show_docs_page)
        docs_button.pack(pady=20)
    
    def setup_dashboard_page(self):
        """Setup the dashboard page"""
        back_button = ttk.Button(self.dashboard_frame, text="← Back", 
                                command=self.show_main_page)
        back_button.pack(anchor='nw', padx=10, pady=10)
        
        top_frame = ttk.Frame(self.dashboard_frame)
        top_frame.pack(pady=10)
        
        title = ttk.Label(top_frame, text="Dashboard", 
                         font=('Arial', 14, 'bold'))
        title.pack(side=tk.LEFT, padx=10)
        
        refresh_button = ttk.Button(top_frame, text="Refresh", 
                                    command=self.load_dashboard)
        refresh_button.pack(side=tk.LEFT, padx=10)
        
        # Create canvas for charts
        self.dashboard_canvas_frame = ttk.Frame(self.dashboard_frame)
        self.dashboard_canvas_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

    def setup_config_page(self):
        """Setup the configuration page"""
        back_button = ttk.Button(self.config_frame, text="← Back", 
                                command=self.show_main_page)
        back_button.pack(anchor='nw', padx=10, pady=10)
        
        title = ttk.Label(self.config_frame, text="Configure Models", 
                        font=('Arial', 14, 'bold'))
        title.pack(pady=10)
        
        # Model selection
        model_frame = ttk.Frame(self.config_frame)
        model_frame.pack(pady=10, padx=20, fill=tk.X)
        
        ttk.Label(model_frame, text="Select Model:", font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="Analysis Model")
        model_dropdown = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                    values=["Analysis Model", "Poem Model"], 
                                    state='readonly', width=20)
        model_dropdown.pack(side=tk.LEFT, padx=5)
        model_dropdown.bind('<<ComboboxSelected>>', lambda e: self.load_model_config())
        
        load_button = ttk.Button(model_frame, text="Load Config", 
                                command=self.load_model_config)
        load_button.pack(side=tk.LEFT, padx=5)
        
        # Scrollable config form
        canvas = tk.Canvas(self.config_frame)
        scrollbar = ttk.Scrollbar(self.config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Configuration fields
        self.config_entries = {}
        fields = [
            ('name', 'Name'),
            ('description', 'Description'),
            ('purpose', 'Purpose'),
            ('expertise', 'Expertise'),
            ('capabilities', 'Capabilities'),
            ('restrictions', 'Restrictions'),
            ('additional', 'Additional'),
            ('llm', 'LLM Model'),
            ('temperature', 'Temperature (0-1)')
        ]
        
        for field, label in fields:
            field_frame = ttk.Frame(scrollable_frame)
            field_frame.pack(fill=tk.X, padx=20, pady=5)
            
            ttk.Label(field_frame, text=f"{label}:", font=('Arial', 10, 'bold')).pack(anchor='w')
            
            if field in ['description', 'purpose', 'expertise', 'capabilities', 'restrictions', 'additional']:
                entry = scrolledtext.ScrolledText(field_frame, height=4, wrap=tk.WORD, width=70)
                entry.configure(font=('Arial', 10))
                # Make text justify
                entry.tag_configure("justify", justify='left')
                entry.tag_add("justify", "1.0", "end")
            else:
                entry = ttk.Entry(field_frame, width=70)
            
            entry.pack(fill=tk.X, pady=2)
            self.config_entries[field] = entry
        
        canvas.pack(side="left", fill="both", expand=True, padx=20, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Save button
        save_button = ttk.Button(self.config_frame, text="Save Configuration", 
                                command=self.save_model_config)
        save_button.pack(pady=20)
        
        self.config_status = ttk.Label(self.config_frame, text="", font=('Arial', 10))
        self.config_status.pack()

    def load_model_config(self):
        """Load current model configuration"""
        try:
            model_key = '8a79f5ec-c5dd-448c-9611-99610792a04b' if self.model_var.get() == "Analysis Model" else '959058ad-4417-49e3-9e71-252ee2fb033d'
            
            model = LLMModel(model_key)

            try:
                init_response = model.initialize()
                print(f"Init response: {init_response}")
            except Exception as init_error:
                print(f"Initialize error: {init_error}")
                import traceback
                traceback.print_exc()
                raise
            model.initialize()
            
            # Clear and populate fields
            for field, entry in self.config_entries.items():
                if isinstance(entry, scrolledtext.ScrolledText):
                    entry.delete('1.0', tk.END)
                    value = getattr(model, f'_{field}', '')
                    if value:
                        entry.insert('1.0', value)
                else:
                    entry.delete(0, tk.END)
                    value = getattr(model, f'_{field}', '')
                    if value:
                        entry.insert(0, str(value))

            if isinstance(entry, scrolledtext.ScrolledText):
                entry.delete('1.0', tk.END)
                value = getattr(model, f'_{field}', '')
                if value:
                    entry.insert('1.0', value)
                    entry.tag_add("justify", "1.0", "end")  # <-- Add this line here
            else:
                entry.delete(0, tk.END)
                value = getattr(model, f'_{field}', '')
                if value:
                    entry.insert(0, str(value))
            
            self.config_status.config(text="Configuration loaded successfully", foreground="green")
        except Exception as e:
            self.config_status.config(text=f"Error loading config: {e}", foreground="red")
            print(f"Error loading model config: {e}")

    def save_model_config(self):
        """Save model configuration"""
        try:
            model_key = '8a79f5ec-c5dd-448c-9611-99610792a04b' if self.model_var.get() == "Analysis Model" else '959058ad-4417-49e3-9e71-252ee2fb033d'
            
            model = LLMModel(model_key)
            model.initialize()
            
            # Get values from entries
            config = {}
            for field, entry in self.config_entries.items():
                if isinstance(entry, scrolledtext.ScrolledText):
                    value = entry.get('1.0', tk.END).strip()
                else:
                    value = entry.get().strip()
                
                if value:
                    if field == 'temperature':
                        config[field] = float(value)
                    else:
                        config[field] = value
            
            # Update assistant
            response = model.update_assistant(**config)
            
            if response.get('status') == 'Success':
                self.config_status.config(text="Configuration saved successfully!", foreground="green")
                messagebox.showinfo("Success", "Model configuration updated successfully!")
            else:
                self.config_status.config(text="Failed to save configuration", foreground="red")
        except Exception as e:
            self.config_status.config(text=f"Error: {e}", foreground="red")
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
            print(f"Error saving model config: {e}")
    
    def load_dashboard(self):
        """Load and display dashboard visualizations"""
        # Clear previous charts
        for widget in self.dashboard_canvas_frame.winfo_children():
            widget.destroy()
        
        # Create figure with subplots
        fig = Figure(figsize=(12, 8))
        
        # Get ElevenLabs credits
        credits_data = self.get_elevenlabs_credits()
        
        # Get label distribution
        labels_data = self.get_label_distribution()
        
        # ElevenLabs Credits Pie Chart and Statistics (top)
        ax1 = fig.add_subplot(2, 1, 1)
        if credits_data:
            used = credits_data['used']
            remaining = credits_data['remaining']
            colors = ['#ff6b6b', '#4ecdc4']
            ax1.pie([used, remaining], labels=['Used', 'Remaining'], 
                autopct='%1.1f%%', startangle=90, colors=colors)
            
            # Get statistics
            stats = self.get_credits_statistics()
            if stats:
                if stats and stats['num_calls'] > 0:
                    avg_credits = stats['avg_per_call']
                    future_calls = int(remaining / avg_credits) if avg_credits > 0 else 0
                    
                    # Add notification for low credits
                    if future_calls < 10:
                        messagebox.showwarning("Low Credits Warning", 
                                            f"Only {future_calls} TTS calls remaining!\nConsider refilling your ElevenLabs credits.")
                    
                    title_text = f'ElevenLabs Credits\n({remaining}/{credits_data["total"]})\nAvg per call: {avg_credits:.0f} | Remaining calls (estimate): {future_calls}'
            else:
                title_text = f'ElevenLabs Credits\n(Total: {credits_data["total"]})'
            
            ax1.set_title(title_text, fontsize=12, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'ElevenLabs data not available', 
                    ha='center', va='center')
            ax1.set_title('ElevenLabs Credits', fontsize=12, fontweight='bold')
                
        # Label Distribution Column Chart (bottom)
        ax2 = fig.add_subplot(2, 1, 2)
        if labels_data:
            labels = list(labels_data.keys())
            counts = list(labels_data.values())
            bars = ax2.bar(labels, counts, color='#4ecdc4')
            ax2.set_xlabel('Labels', fontweight='bold')
            ax2.set_ylabel('Count', fontweight='bold')
            ax2.set_title('Label Distribution', fontsize=12, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'Label data not available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Label Distribution', fontsize=12, fontweight='bold')
        
        fig.tight_layout(pad=3.0)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.dashboard_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def get_elevenlabs_credits(self):
        """Get ElevenLabs credits information"""
        try:
            # Try to import and get credits from ElevenLabs API
            from elevenlabs import ElevenLabs
            
            with open('.env', 'r') as env:
                api_key = env.readlines()[0].split('=')[1].strip()
                print("API key", api_key)
            client = ElevenLabs(api_key=api_key)
            user_info = client.user.get()
            subscription = user_info.subscription

            total = subscription.character_limit
            used = subscription.character_count
            remaining = total - used

            return {
                'total': total,
                'used': used,
                'remaining': remaining
            }
        except Exception as e:
            print(f"Error getting ElevenLabs credits: {e}")
            # Return sample data if API not available
            return {
                'total': 10000,
                'used': 3500,
                'remaining': 6500
            }
        
    def get_credits_statistics(self):
        """Calculate average credits per call and future calls possible"""
        try:
            current_dir = Path(__file__).parent
            credits_csv = current_dir.parent / "credits.csv"
            
            print(f"Looking for credits.csv at: {credits_csv}")
            
            if not credits_csv.exists():
                print("credits.csv not found")
                return None
            
            credits_used_list = []
            with open(credits_csv, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(';')
                    print(f"Parsed line: {parts}")
                    if len(parts) >= 6:
                        used = int(parts[4])  # Used credits
                        credits_used_list.append(used)
            
            print(f"Credits used list: {credits_used_list}")
            
            if not credits_used_list or len(credits_used_list) < 2:
                print(f"Not enough data: {len(credits_used_list)} entries")
                return None
            
            # Calculate differences between consecutive entries
            credits_per_call = []
            for i in range(1, len(credits_used_list)):
                diff = credits_used_list[i] - credits_used_list[i-1]
                if diff > 0:  # Only count increases
                    credits_per_call.append(diff)
            
            print(f"Credits per call: {credits_per_call}")
            
            if not credits_per_call:
                print("No credits per call data")
                return None
            
            avg_credits = sum(credits_per_call) / len(credits_per_call)
            print(f"Average credits: {avg_credits}")
            return {
                'avg_per_call': avg_credits,
                'num_calls': len(credits_per_call)
            }
        except Exception as e:
            print(f"Error reading credits statistics: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_label_distribution(self):
        """Get label distribution from CSV file"""
        try:
            current_dir = Path(__file__).parent
            # Adjust the path to your labels CSV file
            labels_csv = current_dir.parent / "predictions.csv"
            
            if not labels_csv.exists():
                print(f"Labels CSV not found at {labels_csv}")
                return None
            
            labels = []
            with open(labels_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';', fieldnames=['datetime', 'label'])
                for row in reader:
                    label = row.get('label')
                    if label:
                        labels.append(label)
            
            # Count occurrences
            label_counts = Counter(labels)
            return dict(label_counts)
        except Exception as e:
            print(f"Error reading labels: {e}")
            return None
    
    def setup_manual_page(self):
        """Setup the manual page"""
        back_button = ttk.Button(self.manual_frame, text="← Back", 
                                command=self.show_main_page)
        back_button.pack(anchor='nw', padx=10, pady=10)
        
        title = ttk.Label(self.manual_frame, text="User Manual", 
                         font=('Arial', 14, 'bold'))
        title.pack(pady=10)
        
        self.manual_text = scrolledtext.ScrolledText(self.manual_frame, 
                                                     wrap=tk.WORD, 
                                                     width=80, 
                                                     height=30)
        self.manual_text.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        self.load_manual()
    
    def setup_contact_page(self):
        """Setup the contact page"""
        back_button = ttk.Button(self.contact_frame, text="← Back", 
                                command=self.show_main_page)
        back_button.pack(anchor='nw', padx=10, pady=10)
        
        title = ttk.Label(self.contact_frame, text="Contact Information", 
                         font=('Arial', 14, 'bold'))
        title.pack(pady=10)
        
        self.contact_text = scrolledtext.ScrolledText(self.contact_frame, 
                                                      wrap=tk.WORD, 
                                                      width=80, 
                                                      height=30)
        self.contact_text.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        self.load_contact()
    
    def setup_logs_page(self):
        """Setup the logs page"""
        back_button = ttk.Button(self.logs_frame, text="← Back", 
                                command=self.show_main_page)
        back_button.pack(anchor='nw', padx=10, pady=10)
        
        top_frame = ttk.Frame(self.logs_frame)
        top_frame.pack(pady=10)
        
        title = ttk.Label(top_frame, text="Installation Logs", 
                         font=('Arial', 14, 'bold'))
        title.pack(side=tk.LEFT, padx=10)
        
        refresh_button = ttk.Button(top_frame, text="Refresh", 
                                    command=self.load_logs)
        refresh_button.pack(side=tk.LEFT, padx=10)
        
        self.logs_text = scrolledtext.ScrolledText(self.logs_frame, 
                                                   wrap=tk.WORD, 
                                                   width=80, 
                                                   height=30)
        self.logs_text.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
    
    def setup_docs_page(self):
        """Setup the documents page"""
        back_button = ttk.Button(self.docs_frame, text="← Back", 
                                command=self.show_main_page)
        back_button.pack(anchor='nw', padx=10, pady=10)
        
        top_frame = ttk.Frame(self.docs_frame)
        top_frame.pack(pady=10)
        
        title = ttk.Label(top_frame, text="Documents", 
                         font=('Arial', 14, 'bold'))
        title.pack(side=tk.LEFT, padx=10)
        
        refresh_button = ttk.Button(top_frame, text="Refresh", 
                                    command=self.load_docs)
        refresh_button.pack(side=tk.LEFT, padx=10)
        
        delete_button = ttk.Button(top_frame, text="Delete All Prints", 
                                   command=self.delete_all_prints)
        delete_button.pack(side=tk.LEFT, padx=10)
        
        info_label = ttk.Label(self.docs_frame, text="Click on a document to open it", 
                              font=('Arial', 10, 'italic'))
        info_label.pack(pady=5)
        
        list_frame = ttk.Frame(self.docs_frame)
        list_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.docs_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, 
                                       font=('Arial', 11), height=25)
        self.docs_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.docs_listbox.yview)
        
        self.docs_listbox.bind('<Double-Button-1>', self.open_selected_doc)
        
        self.docs_files = []
    
    def load_manual(self):
        """Load user manual from file"""
        try:
            current_dir = Path(__file__).parent
            manual_path = current_dir / "USER_MANUAL.md"
            
            with open(manual_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.manual_text.insert('1.0', content)
                self.manual_text.config(state='disabled')
        except Exception as e:
            self.manual_text.insert('1.0', f"Error loading user manual: {e}")
            self.manual_text.config(state='disabled')
    
    def load_contact(self):
        """Load contact information from file"""
        try:
            current_dir = Path(__file__).parent
            contact_path = current_dir / "contact.md"
            
            with open(contact_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.contact_text.insert('1.0', content)
                self.contact_text.config(state='disabled')
        except Exception as e:
            self.contact_text.insert('1.0', f"Error loading contact information: {e}")
            self.contact_text.config(state='disabled')
    
    def load_logs(self):
        """Load logs from file"""
        try:
            current_dir = Path(__file__).parent
            logs_path = current_dir.parent / "installation.log"
            
            self.logs_text.config(state='normal')
            self.logs_text.delete('1.0', tk.END)
            
            with open(logs_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.logs_text.insert('1.0', content)
                self.logs_text.see(tk.END)
                self.logs_text.config(state='disabled')
        except Exception as e:
            self.logs_text.insert('1.0', f"Error loading logs: {e}")
            self.logs_text.config(state='disabled')
    
    def load_docs(self):
        """Load list of documents from docs folder"""
        try:
            current_dir = Path(__file__).parent
            docs_path = current_dir.parent / "docs"
            
            self.docs_listbox.delete(0, tk.END)
            self.docs_files = []
            
            if not docs_path.exists():
                self.docs_listbox.insert(tk.END, "docs folder not found")
                return
            
            files = sorted(docs_path.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            
            if not files:
                self.docs_listbox.insert(tk.END, "No documents found")
                return
            
            for file in files:
                if file.is_file():
                    self.docs_listbox.insert(tk.END, file.name)
                    self.docs_files.append(file)
        except Exception as e:
            self.docs_listbox.insert(tk.END, f"Error loading documents: {e}")
    
    def open_selected_doc(self, event):
        """Open the selected document with default PDF reader"""
        selection = self.docs_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        if index >= len(self.docs_files):
            return
        
        file_path = self.docs_files[index]
        
        try:
            if platform.system() == 'Windows':
                subprocess.Popen(['start', '', str(file_path)], shell=True)
            elif platform.system() == 'Darwin':
                subprocess.Popen(['open', str(file_path)])
            else:
                subprocess.Popen(['xdg-open', str(file_path)])
        except Exception as e:
            print(f"Error opening document: {e}")
    
    def manual_deletion(self):
        """Delete all print files"""
        current_dir = Path(__file__).parent
        prints_location = current_dir.parent / 'docs'
        
        deleted_count = 0
        try:
            for file in os.listdir(prints_location):
                if 'print_' in file:
                    os.remove(prints_location / file)
                    deleted_count += 1
            return deleted_count
        except Exception as e:
            print(f"Error during manual deletion: {e}")
            return 0
    
    def delete_all_prints(self):
        """Button callback to delete all prints"""
        result = messagebox.askyesno("Confirm Deletion", 
                                     "Are you sure you want to delete all print files?")
        if result:
            deleted_count = self.manual_deletion()
            messagebox.showinfo("Deletion Complete", 
                              f"Deleted {deleted_count} print file(s)")
            self.load_docs()
    
    def automatic_deletion(self):
        """Automatically delete prints from today at 23:59"""
        current_dir = Path(__file__).parent
        prints_location = current_dir.parent / 'docs'
        
        date = int(str(datetime.now().date()).replace('-', ''))
        time_str = str(datetime.now().time())[:5].replace(':', '')
        time_int = int(time_str)
        
        if time_int == 2359:
            try:
                for file in os.listdir(prints_location):
                    if f'print_{date}' in file:
                        os.remove(prints_location / file)
                        print(f"Auto-deleted: {file}")
            except Exception as e:
                print(f"Error during automatic deletion: {e}")
    
    def start_automatic_deletion(self):
        """Start background thread for automatic deletion"""
        def deletion_loop():
            while True:
                self.automatic_deletion()
                time.sleep(60)
        
        deletion_thread = threading.Thread(target=deletion_loop, daemon=True)
        deletion_thread.start()
    
    def show_main_page(self):
        """Show main page"""
        self.manual_frame.pack_forget()
        self.contact_frame.pack_forget()
        self.logs_frame.pack_forget()
        self.docs_frame.pack_forget()
        self.dashboard_frame.pack_forget()
        self.config_frame.pack_forget()
        self.main_frame.pack(fill=tk.BOTH, expand=True)
    
    def show_dashboard_page(self):
        """Show dashboard page"""
        self.main_frame.pack_forget()
        self.manual_frame.pack_forget()
        self.contact_frame.pack_forget()
        self.logs_frame.pack_forget()
        self.docs_frame.pack_forget()
        self.config_frame.pack_forget()
        self.dashboard_frame.pack(fill=tk.BOTH, expand=True)
        self.load_dashboard()
    
    def show_manual_page(self):
        """Show manual page"""
        self.main_frame.pack_forget()
        self.contact_frame.pack_forget()
        self.logs_frame.pack_forget()
        self.docs_frame.pack_forget()
        self.dashboard_frame.pack_forget()
        self.config_frame.pack_forget()
        self.manual_frame.pack(fill=tk.BOTH, expand=True)
    
    def show_contact_page(self):
        """Show contact page"""
        self.main_frame.pack_forget()
        self.manual_frame.pack_forget()
        self.logs_frame.pack_forget()
        self.docs_frame.pack_forget()
        self.dashboard_frame.pack_forget()
        self.config_frame.pack_forget()
        self.contact_frame.pack(fill=tk.BOTH, expand=True)
    
    def show_logs_page(self):
        """Show logs page"""
        self.main_frame.pack_forget()
        self.manual_frame.pack_forget()
        self.contact_frame.pack_forget()
        self.docs_frame.pack_forget()
        self.dashboard_frame.pack_forget()
        self.config_frame.pack_forget()
        self.logs_frame.pack(fill=tk.BOTH, expand=True)
        self.load_logs()
    
    def show_docs_page(self):
        """Show docs page"""
        self.main_frame.pack_forget()
        self.manual_frame.pack_forget()
        self.contact_frame.pack_forget()
        self.logs_frame.pack_forget()
        self.dashboard_frame.pack_forget()
        self.config_frame.pack_forget()
        self.docs_frame.pack(fill=tk.BOTH, expand=True)
        self.load_docs()

    def show_config_page(self):
        """Show config page"""
        self.main_frame.pack_forget()
        self.manual_frame.pack_forget()
        self.contact_frame.pack_forget()
        self.logs_frame.pack_forget()
        self.docs_frame.pack_forget()
        self.dashboard_frame.pack_forget()
        self.config_frame.pack(fill=tk.BOTH, expand=True)
        self.load_model_config()

if __name__ == "__main__":
    root = tk.Tk()
    app = ArtInstallationApp(root)
    root.mainloop()