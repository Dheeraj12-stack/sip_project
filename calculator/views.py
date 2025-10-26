from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io, base64

def index(request):
    return render(request, 'calculator/index.html')

def calculate_sip(request):
    if request.method == 'POST':
        monthly_investment = float(request.POST['investment'])
        annual_rate = float(request.POST['rate'])
        years = int(request.POST['years'])
        
        months = years * 12
        monthly_rate = annual_rate / 12 / 100

        # SIP calculation using pandas
        df = pd.DataFrame({'Month': range(1, months+1)})
        df['Investment'] = monthly_investment
        df['Value'] = df['Month'].apply(lambda m: monthly_investment * (((1 + monthly_rate)**m - 1) / monthly_rate) * (1 + monthly_rate))
        
        total_investment = monthly_investment * months
        future_value = df['Value'].iloc[-1]
        gain = future_value - total_investment

        # AI Prediction using Linear Regression
        model = LinearRegression()
        X = df[['Month']]
        y = df['Value']
        model.fit(X, y)
        future_months = np.array([[months + 12]])
        predicted_value = model.predict(future_months)[0]

        # Matplotlib chart
        plt.figure(figsize=(6, 4))
        plt.plot(df['Month'], df['Value'], color='darkblue', linewidth=2)
        plt.title('SIP Growth Over Time', fontsize=14)
        plt.xlabel('Months')
        plt.ylabel('Investment Value (₹)')
        plt.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        graph = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        context = {
            'investment': monthly_investment,
            'rate': annual_rate,
            'years': years,
            'total_investment': round(total_investment, 2),
            'future_value': round(future_value, 2),
            'gain': round(gain, 2),
            'predicted_value': round(predicted_value, 2),
            'graph': graph,
        }
        return render(request, 'calculator/result.html', context)
    return render(request, 'calculator/index.html')
