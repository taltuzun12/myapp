from flask import Flask, render_template
import psycopg2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Set up database connection
conn = psycopg2.connect(
    host="127.0.0.1",
    port=5432,
    database="postgres",
    user="postgres",
    password="1234"
)

# Define a route to display the data and the model prediction
@app.route('/')
def index():
    # Get data from the database
    cur = conn.cursor()
    cur.execute("SELECT id, age FROM your_table1")
    data = cur.fetchall()

    # Calculate age sum using NumPy
    data = np.array(data)
    age_sum = int(np.sum(data[:, 1]))

    # Define a simple TensorFlow model
    x = tf.constant(data[:, 1], dtype=tf.float32)
    y = 2*x + 1
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer=tf.keras.optimizers.Adam(1), loss='mean_squared_error')
    model.fit(x, y, epochs=10)

    # Use the model to predict a value
    prediction = model.predict([30])

    # Render the data and the prediction in a template
    return render_template('index.html', data=data, age_sum=age_sum, prediction=prediction[0][0])

if __name__ == '__main__':
    app.run(debug=True)
