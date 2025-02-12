import numpy as np
import matplotlib.pyplot as plt

# Data generation function
def generate_data(num_points, num_inliers, num_outliers):
    
    # Adding inliers
    x_in = np.random.rand(num_inliers) * 4000
    y_in = 2 * x_in + 200 + np.random.normal(0, 1, num_inliers)
    
    # Adding outliers
    x_out = np.random.rand(num_outliers) * 4000
    y_out = 2 * x_out + 200 + 500 * np.random.normal(0, 1, num_outliers)
    
    x = np.concatenate((x_in, x_out))
    y = np.concatenate((y_in, y_out))
    
    xy = np.concatenate((x.reshape((len(x),1)), y.reshape((len(y),1))), axis=1)
    
    # Shuffling the indices of the inliers and outliers
    np.random.shuffle(xy)
    xy = np.array(xy)
    x = xy[:,0]
    y = xy[:,1]
    return x, y


# Main function
if __name__ == "__main__":

    num_points = 300
    outliers_ratio = [0.1, 0.3, 0.5, 0.7]

    for outlier_ratio in outliers_ratio:
        
        num_outliers = int(300 * outlier_ratio)
        num_inliers = num_points - num_outliers
        x, y = generate_data(num_points, num_inliers, num_outliers)

        # RANSAC implementation
        num_iterations = 1000
        max_distance = 2

        best_model = None
        best_inliers = None
        best_num_inliers = 0

        for _ in range(num_iterations):
            # Choosing 2 random points
            random_indices = np.random.choice(num_points, 2, replace=False)
            x1, x2 = x[random_indices]
            y1, y2 = y[random_indices]
            
            # Model parameters computation
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            
            y_pred = m * x + b
            
            # Computing distance
            distances = np.abs(y - y_pred)
            
            # Computing numbers of inliers depending on threshold
            inliers = np.where(distances < max_distance)
            num_inliers = len(inliers[0])
            
            # Updating the model
            if num_inliers > best_num_inliers:
                best_model = (m, b)
                best_inliers = inliers
                best_num_inliers = num_inliers

        best_m, best_b = best_model

        # Visualization
        plt.scatter(x, y, c='b', label='Total Data')
        plt.scatter(x[best_inliers], y[best_inliers], c='r', label='Inliers')
        plt.plot(x, best_m * x + best_b, c='g', label='Best Fit')
        plt.legend()
        plt.show()

        print('Slope : {}, intercept : {}'.format(best_m, best_b))
