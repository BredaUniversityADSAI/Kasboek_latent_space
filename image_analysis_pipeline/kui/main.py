import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pathlib import Path
import webbrowser
import platform
import subprocess
import threading
import time
import os
from datetime import datetime

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
        
        # Setup pages
        self.setup_main_page()
        self.setup_manual_page()
        self.setup_contact_page()
        self.setup_logs_page()
        self.setup_docs_page()
        
        # Start automatic deletion thread
        self.start_automatic_deletion()
        
        # Show main page initially
        self.show_main_page()
    
    def setup_main_page(self):
        """Setup the main page"""
        title = ttk.Label(self.main_frame, text="Kasboek: Latent Space, Through the Eyes of the Algorithm", 
                         font=('Arial', 16, 'bold'))
        title.pack(pady=50)
        
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
            manual_path = current_dir / "README.md"
            
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
            self.load_docs()  # Refresh the list
    
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
                time.sleep(60)  # Check every minute
        
        deletion_thread = threading.Thread(target=deletion_loop, daemon=True)
        deletion_thread.start()
    
    def show_main_page(self):
        """Show main page"""
        self.manual_frame.pack_forget()
        self.contact_frame.pack_forget()
        self.logs_frame.pack_forget()
        self.docs_frame.pack_forget()
        self.main_frame.pack(fill=tk.BOTH, expand=True)
    
    def show_manual_page(self):
        """Show manual page"""
        self.main_frame.pack_forget()
        self.contact_frame.pack_forget()
        self.logs_frame.pack_forget()
        self.docs_frame.pack_forget()
        self.manual_frame.pack(fill=tk.BOTH, expand=True)
    
    def show_contact_page(self):
        """Show contact page"""
        self.main_frame.pack_forget()
        self.manual_frame.pack_forget()
        self.logs_frame.pack_forget()
        self.docs_frame.pack_forget()
        self.contact_frame.pack(fill=tk.BOTH, expand=True)
    
    def show_logs_page(self):
        """Show logs page"""
        self.main_frame.pack_forget()
        self.manual_frame.pack_forget()
        self.contact_frame.pack_forget()
        self.docs_frame.pack_forget()
        self.logs_frame.pack(fill=tk.BOTH, expand=True)
        self.load_logs()
    
    def show_docs_page(self):
        """Show docs page"""
        self.main_frame.pack_forget()
        self.manual_frame.pack_forget()
        self.contact_frame.pack_forget()
        self.logs_frame.pack_forget()
        self.docs_frame.pack(fill=tk.BOTH, expand=True)
        self.load_docs()

if __name__ == "__main__":
    root = tk.Tk()
    app = ArtInstallationApp(root)
    root.mainloop()