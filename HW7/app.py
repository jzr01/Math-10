import numpy as np
import pandas as pd
import altair as alt
from sklearn
from sklearn.cluster import KMeans
import streamlit as st

import numpy as np
import pandas as pd
import altair as alt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

 
if __name__ == '__main__':

    st.title('K-Means showcase')

    iterations = st.slider('Choose the number of iteration',1,100,1)

    X, _ = sklearn.datasets.make_blobs(n_samples=1000, centers=5, n_features=2, random_state = 1)
    df = pd.DataFrame(X, columns = list("ab"))
    starting_points = np.array([[0,0],[-2,0],[-4,0],[0,2],[0,4]])
    kmeans = KMeans(n_clusters = 5, max_iter= iterations, init=starting_points, n_init = 1)
    kmeans.fit(X);
    df["c"] = kmeans.predict(X)
    chart = alt.Chart(df).mark_circle().encode(
        x = "a",
        y = "b",
        color = "c:N"
    )    

    
    st.altair_chart(chart)
