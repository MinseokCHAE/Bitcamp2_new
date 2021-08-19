weight = 0.01
input = 1
goal_prediction = 0.8
lr = 0.005
epochs = 234

for iteration in range(epochs):
    prediction = input * weight
    error = (prediction - goal_prediction) **2
    print('error : ' + str(error) + '\tPrediction : ' + str(prediction))

    up_prediction = input * (weight + lr)
    up_error = (goal_prediction - up_prediction) **2

    down_prediction = input * (weight - lr)
    down_error = (goal_prediction - down_prediction) **2

    if(down_error < up_error):
        weight = weight - lr
    if(down_error > up_error):
        weight = weight + lr
