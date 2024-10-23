# # from flask import Flask, request, render_template, jsonify
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from sklearn.cluster import KMeans, AgglomerativeClustering
# # from sklearn.impute import SimpleImputer
# # import threading
# # import io
# # import os

# # app = Flask(__name__)

# # # Function for data clustering (KMeans, Agglomerative, etc.)
# # def perform_clustering(data, algorithm='kmeans', n_clusters=5):
# #     # Keep only numeric columns for clustering
# #     numeric_data = data.select_dtypes(include=['number'])
    
# #     # Impute missing values
# #     imputer = SimpleImputer(strategy='mean')
# #     numeric_data_imputed = imputer.fit_transform(numeric_data)

# #     # Choose the clustering algorithm
# #     if algorithm == 'kmeans':
# #         model = KMeans(n_clusters=n_clusters)
# #     elif algorithm == 'hierarchical':
# #         model = AgglomerativeClustering(n_clusters=n_clusters)
# #     else:
# #         raise ValueError("Unsupported algorithm selected")
    
# #     # Apply clustering and add cluster labels to data
# #     data['Cluster'] = model.fit_predict(numeric_data_imputed)
# #     return data

# # # Function to create visualizations
# # def create_cluster_plot(data):
# #     plt.figure(figsize=(10, 6))
# #     sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue='Cluster', palette='viridis')
# #     plt.title('Gene Clusters')
    
# #     # Save plot to buffer
# #     img_buffer = io.BytesIO()
# #     plt.savefig(img_buffer, format='png')
# #     img_buffer.seek(0)
    
# #     return img_buffer

# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # @app.route('/cluster', methods=['POST'])
# # def cluster():
# #     # Get the file and cluster params from the POST request
# #     file = request.files['file']
# #     algorithm = request.form.get('algorithm', 'kmeans')  # Algorithm: kmeans, hierarchical
# #     n_clusters = int(request.form.get('n_clusters', 5))  # Number of clusters

# #     if not file:
# #         return "No file uploaded", 400
    
# #     # Detect file type and load data accordingly
# #     filename = file.filename
# #     if filename.endswith('.csv'):
# #         data = pd.read_csv(file)
# #     elif filename.endswith('.json'):
# #         data = pd.read_json(file)
# #     elif filename.endswith('.xlsx'):
# #         data = pd.read_excel(file)
# #     else:
# #         return "Unsupported file format", 400
    
# #     # Perform clustering in a separate thread to allow scalability
# #     def process_clustering():
# #         clustered_data = perform_clustering(data, algorithm, n_clusters)
# #         clustered_data.to_csv('clustered_data.csv', index=False)
# #         plot_buffer = create_cluster_plot(clustered_data)
        
# #         # Save plot as file
# #         with open('static/cluster_plot.png', 'wb') as f:
# #             f.write(plot_buffer.read())
    
# #     clustering_thread = threading.Thread(target=process_clustering)
# #     clustering_thread.start()
    
# #     return "Clustering in progress. Check the results soon."

# # @app.route('/results')
# # def results():
# #     # Serve the clustered data and visualization
# #     if os.path.exists('clustered_data.csv') and os.path.exists('static/cluster_plot.png'):
# #         return jsonify({
# #             'message': 'Clustering completed!',
# #             'data_file': 'clustered_data.csv',
# #             'plot_url': '/static/cluster_plot.png'
# #         })
# #     else:
# #         return "No results available yet.", 404

# # if __name__ == '__main__':
# #     app.run(debug=True)




# from flask import Flask, request, render_template, jsonify
# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.impute import SimpleImputer
# import seaborn as sns
# import matplotlib.pyplot as plt
# import threading
# import os

# app = Flask(__name__)

# # Function to perform clustering
# def perform_clustering(data, algorithm='kmeans', n_clusters=5):
#     # Keep only numeric columns for clustering
#     numeric_data = data.select_dtypes(include=['number'])
    
#     # Check if there are any numeric columns to process
#     if numeric_data.empty:
#         raise ValueError("No numeric data available for clustering.")
    
#     # Handle missing values by imputing the mean
#     imputer = SimpleImputer(strategy='mean')
#     numeric_data_imputed = imputer.fit_transform(numeric_data)
    
#     # Choose the clustering algorithm
#     if algorithm == 'kmeans':
#         model = KMeans(n_clusters=n_clusters)
    
#     # Apply clustering
#     clusters = model.fit_predict(numeric_data_imputed)
    
#     # Add clusters to the original data
#     data['Cluster'] = clusters
    
#     return data

# # Function to visualize clusters
# def visualize_clusters(data, output_file='cluster_plot.png'):
#     plt.figure(figsize=(10, 6))
    
#     # Visualize clusters using Seaborn
#     sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue=data['Cluster'], palette='viridis')
    
#     plt.title("Cluster Visualization")
#     plt.savefig(output_file)
#     plt.close()

# # Route for the home page
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Asynchronous clustering and visualization processing
# def process_clustering(data, algorithm, n_clusters):
#     try:
#         # Perform clustering
#         clustered_data = perform_clustering(data, algorithm, n_clusters)
        
#         # Save clustered data
#         clustered_data.to_csv('clustered_data.csv', index=False)
        
#         # Visualize clusters and save the plot
#         visualize_clusters(clustered_data, 'cluster_plot.png')
#     except ValueError as e:
#         print(f"Error during clustering: {e}")

# # Route for clustering
# @app.route('/cluster', methods=['POST'])
# def cluster():
#     # Get the file from the POST request
#     file = request.files['file']
#     if not file:
#         return "No file uploaded", 400
    
#     # Read the file into a dataframe
#     try:
#         data = pd.read_csv(file)
#     except Exception as e:
#         return f"Error reading file: {e}", 400
    
#     # Get clustering parameters from the request
#     n_clusters = int(request.form.get('n_clusters', 5))
#     algorithm = request.form.get('algorithm', 'kmeans')

#     # Run clustering in a separate thread to prevent blocking
#     threading.Thread(target=process_clustering, args=(data, algorithm, n_clusters)).start()
    
#     return jsonify({
#         "message": "Clustering initiated. Check clustered_data.csv and cluster_plot.png for results."
#     })

# if __name__ == '__main__':
#     app.run(debug=True)





# import os
# from flask import Flask, request, render_template, send_file, jsonify
# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.impute import SimpleImputer
# import matplotlib.pyplot as plt
# import seaborn as sns
# import threading

# app = Flask(__name__)

# # Function to perform clustering
# def perform_clustering(data, algorithm='kmeans', n_clusters=5):
#     # Keep only numeric columns for clustering
#     numeric_data = data.select_dtypes(include=['number'])
    
#     # Check if there are any numeric columns to process
#     if numeric_data.empty:
#         raise ValueError("No numeric data available for clustering.")
    
#     # Handle missing values by imputing the mean
#     imputer = SimpleImputer(strategy='mean')
#     numeric_data_imputed = imputer.fit_transform(numeric_data)
    
#     # Choose the clustering algorithm
#     if algorithm == 'kmeans':
#         model = KMeans(n_clusters=n_clusters)
    
#     # Apply clustering
#     clusters = model.fit_predict(numeric_data_imputed)
    
#     # Add clusters to the original data
#     data['Cluster'] = clusters
    
#     return data

# # Function to visualize clusters
# def visualize_clusters(data, output_file='static/cluster_plot.png'):
#     plt.figure(figsize=(10, 6))
    
#     # Visualize clusters using Seaborn
#     sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue=data['Cluster'], palette='viridis')
    
#     plt.title("Cluster Visualization")
    
#     # Ensure the static directory exists
#     if not os.path.exists('static'):
#         os.makedirs('static')
    
#     plt.savefig(output_file)
#     plt.close()

# # Asynchronous clustering and visualization processing
# def process_clustering(data, algorithm, n_clusters):
#     try:
#         # Perform clustering
#         clustered_data = perform_clustering(data, algorithm, n_clusters)
        
#         # Ensure the static directory exists before saving the CSV file
#         if not os.path.exists('static'):
#             os.makedirs('static')
        
#         # Save clustered data
#         clustered_data.to_csv('static/clustered_data.csv', index=False)
        
#         # Visualize clusters and save the plot
#         visualize_clusters(clustered_data, 'static/cluster_plot.png')
#     except ValueError as e:
#         print(f"Error during clustering: {e}")

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/cluster', methods=['POST'])
# def cluster():
#     try:
#         # Get the file from the POST request
#         file = request.files['file']
#         if not file:
#             return "No file uploaded", 400
        
#         # Read the file into a dataframe
#         data = pd.read_csv(file)
        
#         # Get the number of clusters from the user, default to 5 if not provided
#         n_clusters = int(request.form.get('n_clusters', 5))
        
#         # Choose clustering algorithm, default to K-Means
#         algorithm = request.form.get('algorithm', 'kmeans')
        
#         # Run clustering in a separate thread
#         threading.Thread(target=process_clustering, args=(data, algorithm, n_clusters)).start()
        
#         return jsonify({"message": "Clustering initiated. Check clustered_data.csv and cluster_plot.png for results."})
    
#     except Exception as e:
#         return f"An error occurred: {str(e)}", 500

# @app.route('/download/csv')
# def download_csv():
#     csv_path = 'static/clustered_data.csv'
#     if os.path.exists(csv_path):
#         return send_file(csv_path, as_attachment=True)
#     else:
#         return "CSV file not found.", 404

# @app.route('/download/plot')
# def download_plot():
#     plot_path = 'static/cluster_plot.png'
#     if os.path.exists(plot_path):
#         return send_file(plot_path, as_attachment=True)
#     else:
#         return "Plot image not found.", 404

# @app.route('/show_plot')
# def show_plot():
#     plot_path = 'static/cluster_plot.png'
#     if os.path.exists(plot_path):
#         return send_file(plot_path)
#     else:
#         return "Plot image not found.", 404

# if __name__ == '__main__':
#     app.run(debug=True)







from flask import Flask, request, render_template, send_file
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import threading
import os

app = Flask(__name__)

# Create 'static' folder if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

def perform_clustering(data, n_clusters):
    try:
        # Keep only numeric columns
        numeric_data = data.select_dtypes(include=['number'])
        if numeric_data.empty:
            raise ValueError("No numeric data available for clustering.")
        
        # Handle missing values by imputing the mean
        imputer = SimpleImputer(strategy='mean')
        numeric_data_imputed = imputer.fit_transform(numeric_data)

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300)
        cluster_labels = kmeans.fit_predict(numeric_data_imputed)
        data['Cluster'] = cluster_labels

        # Save the clustered data to CSV
        clustered_file_path = 'static/clustered_data.csv'
        data.to_csv(clustered_file_path, index=False)

        # Plotting with Seaborn for visualizing clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=numeric_data_imputed[:, 0], y=numeric_data_imputed[:, 1], hue=cluster_labels, palette='viridis')
        plt.title(f'Clustering with {n_clusters} Clusters')
        plt.savefig('static/cluster_plot.png')
        plt.close()

        return clustered_file_path, 'static/cluster_plot.png'
    except Exception as e:
        print(f"Error during clustering: {e}")
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    try:
        # Get the uploaded file
        file = request.files['file']
        if not file:
            return "No file uploaded", 400

        # Read file based on extension
        filename = file.filename
        if filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif filename.endswith('.xlsx'):
            data = pd.read_excel(file)
        elif filename.endswith('.json'):
            data = pd.read_json(file)
        else:
            return "Unsupported file format", 400

        # Get clustering parameters from the form
        n_clusters = int(request.form.get('n_clusters', 5))

        # Run clustering in a separate thread
        threading.Thread(target=process_clustering, args=(data, n_clusters)).start()

        return {
            "message": "Clustering initiated. Check clustered_data.csv and cluster_plot.png for results."
        }
    except Exception as e:
        return f"Error: {e}", 500

def process_clustering(data, n_clusters):
    clustered_file, plot_file = perform_clustering(data, n_clusters)
    if not clustered_file or not plot_file:
        return "Error during clustering."

@app.route('/download/csv')
def download_csv():
    return send_file('static/clustered_data.csv', as_attachment=True)

@app.route('/download/plot')
def download_plot():
    return send_file('static/cluster_plot.png', as_attachment=True)

@app.route('/show_plot')
def show_plot():
    return render_template('show_plot.html')

if __name__ == '__main__':
    app.run(debug=True)
