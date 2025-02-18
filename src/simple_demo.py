import numpy as np
import pandas as pd
from models import *
from utils import auxiliary_plots, metrics
from preprocessing import normalization, data_generation
import time, datetime

TRAIN_FILE_NAME = 'data/hourly_20140102_20191101_train.csv'
TEST_FILE_NAME = 'data/hourly_20140102_20191101_test.csv'

RESULT_FILE_NAME = 'files/results/simple_demo_result.csv'

FORECAST_HORIZON = 24
PAST_HISTORY = 288

BATCH_SIZE = 100
BUFFER_SIZE = 10000

EPOCHS = 50

res = pd.DataFrame(
    columns=['Seed', 'BS', 'History', 'WAPE']
)

for SEED in range(10):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    SHOW_PLOTS = False

    # Read train file
    with open(TRAIN_FILE_NAME, 'r') as datafile:
        ts_train = datafile.readlines()[1:]  # skip the header
        ts_train = np.asarray([np.asarray(l.rstrip().split(',')[0], dtype=np.float32) for l in ts_train])
        ts_train = np.reshape(ts_train, (ts_train.shape[0],))

    # Read test data file
    with open(TEST_FILE_NAME, 'r') as datafile:
        ts_test = datafile.readlines()[1:]  # skip the header
        ts_test = np.asarray([np.asarray(l.rstrip().split(',')[0], dtype=np.float32) for l in ts_test])
        ts_test = np.reshape(ts_test, (ts_test.shape[0],))

    # Train/validation split
    TRAIN_SPLIT = int(ts_train.shape[0]*1)

    # Normalize training data
    norm_params = normalization.get_normalization_params(ts_train[:TRAIN_SPLIT])
    ts_train = normalization.normalize(ts_train, norm_params)
    # Normalize test data with train params
    ts_test = normalization.normalize(ts_test, norm_params)

    # Get x and y for training and validation
    x_train, y_train = data_generation.univariate_data(ts_train, 0, TRAIN_SPLIT, PAST_HISTORY, FORECAST_HORIZON)
    x_val, y_val = data_generation.univariate_data(ts_train, TRAIN_SPLIT-PAST_HISTORY, ts_train.shape[0], PAST_HISTORY, FORECAST_HORIZON)
    # Get x and y for test data
    x_test, y_test = data_generation.univariate_data(ts_test, 0, ts_test.shape[0], PAST_HISTORY, FORECAST_HORIZON)

    # Convert numpy data to tensorflow dataset
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE).repeat()
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)


    # Create model
    # model = simple_lstm(x_train.shape, FORECAST_HORIZON, 'adam', 'mae')
    # model = mlp(x_train.shape, FORECAST_HORIZON, 'adam', 'mae', hidden_layers=[64,32,16])
    # model = cldnn(x_train.shape, FORECAST_HORIZON, 'adam', 'mae')
    # model = cnn(x_train.shape, FORECAST_HORIZON, 'adam', 'mae')
    model = tcn(x_train.shape, FORECAST_HORIZON, 'adam', 'mae')
    ts = str(datetime.datetime.fromtimestamp(time.time()))[5:-7].replace(' ','_').replace(':','-')
    
    print(model.summary())

    evaluation_interval = int(np.ceil(x_train.shape[0] / BATCH_SIZE))
    
    #save_best = tf.keras.callbacks.ModelCheckpoint(filepath=f'files/models/best-model-bs-{BATCH_SIZE}-ph-{PAST_HISTORY}-seed-{SEED}-{ts}.ckpt',save_best_only=True, monitor='loss')
                

    # Train the model
    history = model.fit(train_data, epochs=EPOCHS,
                        #callbacks=[save_best],
                        steps_per_epoch=evaluation_interval,
                        #validation_data=val_data, validation_steps=evaluation_interval
                        )

    # Plot training and evaluation loss evolution
    if SHOW_PLOTS:
        auxiliary_plots.plot_training_history(history, ['loss'])

    # Get validation results
    #val_forecast = model.predict(x_val)
    #val_forecast = normalization.denormalize(val_forecast, norm_params)
    #y_val = normalization.denormalize(y_val, norm_params)

    #val_metrics = metrics.evaluate_all(y_val, val_forecast)
    #print('Validation metrics', val_metrics)

    # TEST
    # Predict with test data and get results
    test_forecast = model.predict(test_data)

    test_forecast = normalization.denormalize(test_forecast, norm_params)
    y_test = normalization.denormalize(y_test, norm_params)
    x_test = normalization.denormalize(x_test, norm_params)

    test_metrics = metrics.evaluate_all(y_test, test_forecast)
    print('Test scores', test_metrics)
    run_stats = {
        'Seed':SEED,
        'BS':BATCH_SIZE,
        'History':PAST_HISTORY,
        'WAPE':test_metrics['wape']
    }
    res.loc[len(res)] = run_stats
    res.to_csv('./files/results/seeded_res.csv')
    model.save(f'files/models/best-model-bs-{BATCH_SIZE}-ph-{PAST_HISTORY}-seed-{SEED}-{ts}.ckpt')
    # Plot some test predictions
    if SHOW_PLOTS:
        auxiliary_plots.plot_ts_forecasts(x_test, y_test, test_forecast, 1)




