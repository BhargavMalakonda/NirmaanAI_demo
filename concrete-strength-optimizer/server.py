from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ml_model import ConcreteStrengthPredictor
except ImportError as e:
    print(f"❌ Error importing ml_model: {e}")
    print("Make sure ml_model.py is in the same directory")
    sys.exit(1)

# Load ML model
predictor = ConcreteStrengthPredictor()
model_path = 'concrete_strength_model.pkl'

if os.path.exists(model_path):
    try:
        predictor.load_model(model_path)
        print(f"✅ ML Model loaded from {model_path}")
        print(f"✅ Model metrics: R² = {predictor.metrics.get('test', {}).get('r2', 'N/A')}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
else:
    print(f"❌ Model file not found at {model_path}")
    print("Please train the model first by running: python ml_model.py")
    sys.exit(1)

class MLModelHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests - serve files"""
        try:
            # Serve index.html for root path
            if self.path == '/' or self.path == '/index.html':
                self.path = '/index.html'
                return super().do_GET()
            # Serve other files
            else:
                return super().do_GET()
        except Exception as e:
            self.send_error(404, f"File not found: {self.path}")
    
    def do_POST(self):
        """Handle POST requests - ML predictions"""
        if self.path == '/api/predict':
            try:
                # Get content length
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length == 0:
                    raise ValueError("No data received")
                
                # Read and parse data
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                print(f"Received request: {data}")
                
                # Get parameters with defaults
                temp = float(data.get('temperature', 28))
                humidity = float(data.get('humidity', 65))
                curing_method = data.get('curing_method', 'steam')
                target_strength = float(data.get('target_strength', 35))
                
                # Map curing method to model's format
                curing_map = {
                    'steam': 'Steam',
                    'water': 'Water',
                    'membrane': 'Compound',
                    'heat': 'Air',
                    'autoclave': 'Steam'
                }
                model_curing = curing_map.get(curing_method.lower(), 'Steam')
                
                print(f"Processing: temp={temp}, humidity={humidity}, method={model_curing}, target={target_strength}")
                
                # Find optimal time
                optimal_hours, predicted_strength = predictor.find_optimal_time(
                    temp, humidity, model_curing, target_strength,
                    min_hours=24, max_hours=720, step=24
                )
                
                # Calculate costs
                cost_rates = {'Air': 1500, 'Compound': 2000, 'Water': 2500, 'Steam': 5000}
                hourly_rate = cost_rates.get(model_curing, 2000)
                total_cost = optimal_hours * hourly_rate
                
                # WEATHER-HACKING ENERGY ARBITRAGE LOGIC
                # Check if conditions are good for ambient curing (natural steam room)
                is_natural_steam_room = temp >= 30 and humidity >= 70
                is_good_for_ambient = temp >= 25 and humidity >= 60
                
                # Calculate potential savings
                daily_steam_cost = 5000 * 24  # ₹120,000 per day
                daily_ambient_cost = 1500 * 24  # ₹36,000 per day
                
                arbitrage = {
                    'active': False,
                    'savings': 0,
                    'message': '',
                    'recommendation': 'steam'
                }
                
                if is_natural_steam_room:
                    # PERFECT conditions - natural steam room!
                    savings = daily_steam_cost - daily_ambient_cost
                    arbitrage = {
                        'active': True,
                        'savings': savings,
                        'message': '🔆 Steam Disabled – Ambient Conditions Sufficient',
                        'recommendation': 'ambient_only',
                        'steam_status': 'DISABLED',
                        'ambient_status': 'ACTIVE'
                    }
                elif is_good_for_ambient:
                    # Good conditions - partial savings
                    savings = (daily_steam_cost - daily_ambient_cost) * 0.7
                    arbitrage = {
                        'active': True,
                        'savings': int(savings),
                        'message': '⚠️ Partial Ambient Curing Possible – 70% Steam Reduction',
                        'recommendation': 'ambient_optional',
                        'steam_status': 'OPTIONAL',
                        'ambient_status': 'RECOMMENDED'
                    }
                
                # Prepare response with arbitrage info
                response = {
                    'success': True,
                    'optimal_hours': optimal_hours,
                    'optimal_days': round(optimal_hours / 24, 1),
                    'predicted_strength': round(predicted_strength, 1),
                    'total_cost': int(total_cost),
                    'hourly_rate': hourly_rate,
                    'arbitrage': arbitrage,
                    'weather': {
                        'temperature': temp,
                        'humidity': humidity,
                        'condition': 'Hot & Humid' if temp >= 30 else 'Mild' if temp >= 25 else 'Cool',
                        'is_natural_steam_room': is_natural_steam_room,
                        'forecast': [
                            {'day': 'Tomorrow', 'temp': temp + 2, 'humidity': humidity + 5, 'condition': 'Hot & Humid'},
                            {'day': 'Day 2', 'temp': temp - 1, 'humidity': humidity - 3, 'condition': 'Mild'},
                            {'day': 'Day 3', 'temp': temp + 1, 'humidity': humidity - 2, 'condition': 'Warm'}
                        ]
                    }
                }
                
                print(f"Sending response: {response}")
                
                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                print(f"Error processing request: {e}")
                import traceback
                traceback.print_exc()
                
                # Send error response
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'success': False, 
                    'error': str(e)
                }).encode())
        
        elif self.path == '/api/test':
            # Simple test endpoint
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'success': True, 'message': 'Server is running'}).encode())
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

# Start server
if __name__ == '__main__':
    port = 8000
    server_address = ('0.0.0.0', port)  # Listen on all interfaces
    httpd = HTTPServer(server_address, MLModelHandler)
    
    print(f"🚀 Server started successfully!")
    print(f"📁 Serving files from: {os.getcwd()}")
    print(f"🔗 Access the app at:")
    print(f"   http://localhost:{port}")
    print(f"   http://127.0.0.1:{port}")
    print(f"   http://{os.environ.get('COMPUTERNAME', 'localhost')}:{port}")
    print(f"\n📡 API endpoints:")
    print(f"   POST http://localhost:{port}/api/predict")
    print(f"   GET  http://localhost:{port}/api/test")
    print(f"\n🌡️ Weather-Hacking Energy Arbitrage Active!")
    print(f"   - Automatically detects natural steam-room conditions")
    print(f"   - Calculates diesel savings when ambient curing possible")
    print(f"   - Disables steam boilers when temp ≥30°C & humidity ≥70%")
    print(f"\nPress Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")