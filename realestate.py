import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import webbrowser
from threading import Timer
import io
import base64

# Initialize the Dash app
app = dash.Dash(__name__)
model = None  # Placeholder for the model, which will be trained after data upload

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        html.H1("Real Estate Price Prediction", style={'text-align': 'center', 'color': '#007BFF'}),

        # Upload dataset section
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag and Drop or ', html.A('Select a CSV File')]),
                style={
                    'width': '50%',
                    'height': '40px',
                    'lineHeight': '40px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px auto',  # Center the upload box
                    'background-color': '#f9f9f9'
                },
                multiple=False
            ),
        ], style={'text-align': 'center'}),

        # Placeholder for the success message
        html.Div(id='upload-status', style={'text-align': 'center', 'font-size': '18px', 'margin-top': '10px', 'color': '#28a745'}),

        # Prediction input section
        html.Div([
            dcc.Input(id='distance_to_mrt', type='number', placeholder='Distance to MRT Station (meters)',
                      style={'margin': '10px', 'padding': '10px', 'width': '80%'}),
            dcc.Input(id='num_convenience_stores', type='number', placeholder='Number of Convenience Stores',
                      style={'margin': '10px', 'padding': '10px', 'width': '80%'}),
            dcc.Input(id='latitude', type='number', placeholder='Latitude',
                      style={'margin': '10px', 'padding': '10px', 'width': '80%'}),
            dcc.Input(id='longitude', type='number', placeholder='Longitude',
                      style={'margin': '10px', 'padding': '10px', 'width': '80%'}),
            html.Button('Predict Price', id='predict_button', n_clicks=0,
                        style={'margin': '10px', 'padding': '10px', 'background-color': '#007BFF', 'color': 'white', 'width': '80%'}),
        ], style={'text-align': 'center'}),

        # Styled output for prediction without background color
        html.Div(id='prediction_output', style={
            'text-align': 'center',
            'font-size': '24px',
            'margin-top': '20px',
            'color': '#007BFF',
            'width': '60%',
            'margin': '20px auto'
        })
    ], style={'width': '50%', 'margin': '0 auto', 'border': '2px solid #007BFF', 'padding': '20px', 'border-radius': '10px', 'background-color': '#f1f1f1'})
])

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return df

# Callback to handle the file upload and train the model
@app.callback(
    Output('upload-status', 'children'),
    Input('upload-data', 'contents'),
)
def update_output(contents):
    global model
    if contents is not None:
        df = parse_contents(contents)
        # Define features and target
        features = ['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude']
        target = 'House price of unit area'
        
        if set(features).issubset(df.columns) and target in df.columns:
            X = df[features]
            y = df[target]
            # Train the model
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)

            return 'File uploaded and model trained successfully!'
        else:
            return 'Invalid data! Please upload a CSV file with the correct format.'

    return 'Please upload a CSV file.'

# Callback to handle predictions
@app.callback(
    Output('prediction_output', 'children'),
    [Input('predict_button', 'n_clicks')],
    [State('distance_to_mrt', 'value'),
     State('num_convenience_stores', 'value'),
     State('latitude', 'value'),
     State('longitude', 'value')]
)
def update_prediction(n_clicks, distance_to_mrt, num_convenience_stores, latitude, longitude):
    if n_clicks > 0:
        if model is None:
            return 'Please upload a dataset and train the model first.'
        if all(v is not None for v in [distance_to_mrt, num_convenience_stores, latitude, longitude]):
            # Prepare the feature vector
            features = pd.DataFrame([[distance_to_mrt, num_convenience_stores, latitude, longitude]],
                                    columns=['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude'])
            # Predict
            prediction = model.predict(features)[0]
            return f'Predicted House Price of Unit Area: ${prediction:.2f}'
        else:
            return 'Please enter all values to get a prediction'
    return ''

# Function to open Chrome browser
def open_browser():
    chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'  # Path to Chrome
    webbrowser.get(chrome_path).open("http://127.0.0.1:8050/")

# Run the app
if __name__ == '__main__':
    Timer(1, open_browser).start()  # Open Chrome after 1 second
    app.run_server(debug=True)
