import streamlit as st
import pandas as pd
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_rnnlayers, n_outputs):
        super(RNN, self).__init__()
        self.D = n_inputs
        self.M = n_hidden
        self.K = n_outputs
        self.L = n_rnnlayers

        self.rnn = nn.LSTM(
            input_size=self.D,
            hidden_size=self.M,
            num_layers=self.L,
            batch_first=True,

        )

        self.fc = nn.Linear(self.M, self.K)

    def forward(self, X):
        h0 = torch.zeros(self.L, X.size(0), self.M).to(device)
        c0 = torch.zeros(self.L, X.size(0), self.M).to(device)

        out, _ = self.rnn(X, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out

def full_gd(model, criterion, optimizer, X_train, y_train, X_test, Y_test, epochs=1000):
    # Stuff to store
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    st.write('Training Info:')
    for it in range(epochs):
        # zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs1 = model(X_train)
        loss1 = criterion(outputs1, y_train)
        outputs2 = model(X_test)
        loss2 = criterion(outputs2, Y_test)
        # Backward and optimize
        loss1.backward()
        loss2.backward()
        optimizer.step()

        # Save losses
        train_losses[it] = loss1.item()
        test_losses[it] = loss2.item()
        if (it + 1) % 100 == 0:
            st.write(f'Epoch {it + 1}/{epochs}, Train Loss: {loss1.item():.4f}, Test Loss: {loss2.item():.4f}')
    return train_losses, test_losses

def plot_chart_date(date):
    chart = alt.Chart(time_dict[date]).mark_circle().encode(
            y='high',
            x='volume',
            color = 'high',
            tooltip=['Name', 'open', 'high', 'close']
        ).properties(
    title=f'The stock price scatter on {str(date)}',
    width=800,
    height=1500
)
    return chart
def plot_chart_name(name):
    chart = alt.Chart(name_dict[name]).mark_line().encode(
        x='date',
        y='high',
        tooltip=['open', 'high', 'close'],
    ).properties(
    title=f'The stock price change for {str(name)}',
    width=800,
    height=300
)
    return chart

def apply_kmeans(n):
    kmeans = KMeans(n)
    kmeans.fit(df4)
    df_mean['Cluster'] = kmeans.predict(df4)
    chart = alt.Chart(df_mean).mark_circle().encode(
        y="high",
        x="open",
        tooltip=['Name'],
        color="Cluster:N"
    ).properties(
    title=f'Clusatering when k = {n})',
    width=800,
    height=300
)
    return chart

if __name__ == '__main__':
    #title
    st.title("Zhengran Ji's Final Project")

    #data handle
    st.markdown('Introduction:\n'+
                'In this project, I will use the data of S&P 500 price information from 2013 to 2018. The links for the raw data and the github page are as follow.')
    a = 'raw data'
    b = 'https://www.kaggle.com/camnugent/sandp500'
    link = f'[{a}]({b})'
    st.markdown(link, unsafe_allow_html=True)
    df = pd.read_csv('all_stocks_5yr.csv')

    #bad rows
    st.write('bad rows')
    if "bad_row" not in st.session_state:
        st.session_state['bad_row'] = df[df.isna().any(axis = 1)]
    bad_row = st.session_state['bad_row']
    st.dataframe(bad_row)
    df = df[~df.isna().any(axis = 1)]
    name_list = list(pd.unique(df['Name']))
    name_dict1 = {name: df[df['Name'] == name] for name in name_list}
    df = df[df['date'].map(lambda x: '2018' in x or '2017' in x)]
    df_1 = df.copy()
    df_1['date'] = pd.to_datetime(df_1['date'])

    #store data
    name_list = list(pd.unique(df_1['Name']))
    time_list = list(pd.unique(df['date']))
    name_dict = {name: df_1[df_1['Name'] == name] for name in name_list }
    time_dict = {time: df[df['date'] == time] for time in time_list }

    #access by date
    st.subheader('Access the dataset by date')
    st.write('By choosing the date in the selection box, you can see the scatter plot of the stock high price on that day')
    date = st.selectbox('Select a date:',time_list)
    if 'chart_date' not in st.session_state:
        st.session_state['chart_date'] = plot_chart_date(date)
    st.session_state['chart_date'] = plot_chart_date(date)
    st.altair_chart(st.session_state['chart_date'])

    #access by company
    st.subheader('Access the dataset by company')
    st.write("By choosing the company name from the selctction boc, you can see the plot of this company's stock price")
    name = st.selectbox('Select a name:', name_list)
    if 'chart_name' not in st.session_state:
        st.session_state['chart_name'] = plot_chart_name(name)
    st.session_state['chart_name'] = plot_chart_name(name)
    st.altair_chart(st.session_state['chart_name'])

    # apply cluster on the data
    st.subheader('Apply clustering on the dataset')
    df = pd.read_csv('all_stocks_5yr.csv')
    df = df[~df.isna().any(axis=1)]
    df_mean = pd.DataFrame(name_dict['MMM'].mean(axis=0)).T
    key_list = ['MMM']
    for key in name_dict.keys():
        if key != 'MMM':
            key_list.append(key)
            df_mean = pd.concat([df_mean, pd.DataFrame(name_dict[key].mean()).T])
    df_mean['Name'] = key_list

    df2 = df_mean.iloc[:, 0:4]
    scaler = StandardScaler()
    scaler.fit(df2)
    df4 = pd.DataFrame(scaler.transform(df2), columns=df2.columns)
    n = st.slider('Choose the number of clusters',1,10,step = 1)
    if 'chart_kmeans' not in st.session_state:
        st.session_state['chart_kmeans'] = apply_kmeans(n)
    st.session_state['chart_kmeans'] = apply_kmeans(n)
    st.altair_chart(st.session_state['chart_kmeans'])

    #apply LSTM to predict the stock price
    st.subheader("Apply LSTM on the return of the Apple stock")
    st.write("I define a new column in the dataset called 'return' which is calculated by today's close price - yesterday's close price.\n In the this section, I build a Recurrent Neural Network with LSTM layers "
             + "and train it with the return before 2017. After that, I use the model to predioct the return of Apple stcok after 2017 and compare it with the real value.")

    fig, ax = plt.subplots()
    app_data = name_dict1['AAPL']
    app_data['date'] = pd.to_datetime(app_data['date'])
    ax.plot(app_data['date'], app_data['high'])
    plt.ylabel("Price($)")
    plt.xlabel('Year')
    plt.title('Stock Price for Apple Inc.')
    st.pyplot(fig)


    if st.button('Apply LSTM to predict the return of the Apple stock', key=4):

        app_data['pre_close'] = app_data['close'].shift(1)
        app_data['return'] = app_data['close'] - app_data['pre_close']
        app_data = app_data[1:]
        plt.plot(app_data['return'])
        series = np.array(app_data['return']).reshape(-1, 1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler = StandardScaler()
        scaler.fit(series[:len(series) // 2])
        series = scaler.transform(series).flatten()

        T = 20
        D = 1
        X = []
        Y = []
        for t in range(len(series) - T):
            x = series[t:t + T]
            X.append(x)
            y = series[t + T]
            Y.append(y)
        X = np.array(X).reshape(-1, T, 1)
        Y = np.array(Y).reshape(-1, 1)
        N = len(X)



        model = RNN(1, 5, 1, 1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        x_train = torch.from_numpy(X[:-N // 2].astype(np.float32))
        y_train = torch.from_numpy(Y[:-N // 2].astype(np.float32))
        x_test = torch.from_numpy(X[-N // 2:].astype(np.float32))
        y_test = torch.from_numpy(Y[-N // 2:].astype(np.float32))


        train_loss,test_loss = full_gd(model, criterion, optimizer, x_train, y_train, x_test, y_test)
        fig,ax = fig, ax = plt.subplots()
        plt.plot(train_loss,label = 'Train')
        plt.plot(test_loss,label = 'Test')
        ax.legend()
        plt.title('Loss Plot')
        st.pyplot(fig)


        validation_target = Y
        validation_predictions = []

        i = 0

        X_on_device = torch.from_numpy(X.astype(np.float32)).to(device)

        while len(validation_predictions) < len(validation_target):
            input_ = X_on_device[i].reshape(1, T, 1)
            p = model(input_)[0, 0].item()
            i += 1
            validation_predictions.append(p)

        fig, ax = plt.subplots()
        ax.plot(validation_target, 'b',label = 'real')
        ax.plot(validation_predictions, 'r',label = 'predicted')
        plt.ylabel("Return($)")
        ax.legend()
        plt.title('Plot of Prediction for Apple Stock Return after 2017')
        st.pyplot(fig)




