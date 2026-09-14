import http.server
import socketserver
import json
import subprocess

PORT = 5000
current_process = None

class ManagerHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        global current_process
        if self.path == '/start':
            length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(length))
            cmd = data.get('cmd')
            
            if current_process:
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(current_process.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            print(f"\n[Manager] Starting: {cmd}")
            current_process = subprocess.Popen(cmd, shell=True)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
            
        elif self.path == '/stop':
            if current_process:
                print("[Manager] Stopping server...")
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(current_process.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                current_process = None
                
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
            
    def log_message(self, format, *args):
        # Suppress noisy HTTP logs
        pass

print(f"Windows Server Manager listening on 0.0.0.0:{PORT}...")
print("Leave this window open. The WSL script will control it automatically.")

with socketserver.TCPServer(("0.0.0.0", PORT), ManagerHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down manager...")
        if current_process:
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(current_process.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

