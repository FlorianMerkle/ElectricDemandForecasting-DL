import tensorflow as tf
from src.preprocessing import normalization
import time
import numpy as np
import pandas as pd
device='cuda'

norm_params = {'mean': 28505.41, 'std': 4596.946, 'max': 41217.0, 'min': 17714.0}
loss_object = tf.keras.losses.MeanAbsoluteError()

def linf_attack(m, x, y, alpha=0.001, epsilon=100, iterations=100):
    df = pd.DataFrame(columns=['iterations', 'alpha', 'l_inf_norm', 'MAE', 'duration'])
    t0 = time.time()
    adv_input = tf.identity(x)  # Create a copy of the input to apply perturbations
    denorm_input = normalization.denormalize(x, norm_params)
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adv_input)
            # Forward pass to get the model's prediction on the adversarial input
            prediction = m(adv_input)
            # Calculate the loss between the true label and the model's prediction
            loss = loss_object(y, prediction)
        # Get the gradients of the loss with respect to the adversarial input
        gradient = tape.gradient(loss, adv_input)
        # Get the sign of the gradients and scale by epsilon
        perturbation = alpha * tf.sign(gradient)
        # Update the adversarial input by adding the perturbation
        adv_input = adv_input + perturbation
        # Project the adversarial input back into the epsilon-ball
        denorm_adv_input = normalization.denormalize(adv_input, norm_params)        
        denorm_adv_input = tf.clip_by_value(denorm_adv_input, denorm_input - epsilon, denorm_input + epsilon)
        adv_input = normalization.normalize(denorm_adv_input, norm_params)
    mae, avg_mae = calc_denormalized_mae(m, x, adv_input)
    
    np.save(f'./adversarial-examples/linf/iterations-{iterations}-epsilon-{epsilon}-alpha-{alpha}-mae={avg_mae}.npy', (adv_input).numpy())
        
    row={
            'iterations':iterations, 
            'alpha':alpha, 
            'l_inf_norm':epsilon,
            'MAE':mae,
            'duration':t0-time.time()
        }
    df.loc[len(df)] = row
    df.to_pickle(f'./results/linf/iterations-{iterations}-epsilon-{epsilon}-alpha-{alpha}-mae={avg_mae}.pkl')
    perturbation = adv_input - x
    return perturbation, t0-time.time()


def loss_fn(x, adv_x,y, prediction, const):
    MAE = tf.keras.losses.MeanAbsoluteError()
    pert = (adv_x - x)#*w
    l2 = tf.norm(pert, ord='euclidean')
    mae = MAE(y, prediction)
    return l2 - const * mae

def l2_attack_for_l0_attack(x,y,m,mask,iterations, const,lr):
    pert = tf.Variable(tf.zeros_like(x))
    adv_input = tf.Variable(tf.zeros_like(x))
    optimizer = tf.keras.optimizers.Adam(lr)
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(pert)
            adv_input = adv_input + pert * mask
            # Forward pass to get the model's prediction on the adversarial input
            prediction = m(adv_input)
            # Calculate the loss between the true label and the model's prediction
            loss = loss_fn(x,adv_input, y,prediction, const)
            grads = tape.gradient(loss, pert)
            optimizer.apply_gradients([(grads, pert)])
    return pert, grads

def l0_attack(x,y,m,ITER = 24, l2_iter=300, l2_c=5,l2_lr=.03):
    df = pd.DataFrame(columns=['l2_iter', 'l2_lr', 'l2_constant', 'l0_norm', 'MAE', 'duration'])
    t0 = time.time()
    mask = np.ones_like(x)
    
    for _ in range(ITER):
        print(_)
        pert, grads = l2_attack_for_l0_attack(x, y, m, mask, iterations=l2_iter, const=l2_c, lr=l2_lr)
    
        total_abs_change = tf.math.abs(pert * grads)
        non_zeros = np.count_nonzero(mask, axis=1)

        sorted_indices = np.argsort(total_abs_change, axis=1)
        count = np.zeros_like(non_zeros)
        for k in range(mask.shape[0]):
            idx = sorted_indices[k][:-int(non_zeros[k]*.8)]
            non_zero_indices = np.where(mask[k, idx] != 0)[0]
            # Update count and mask
            count[k] += len(non_zero_indices)
            mask[k, idx[non_zero_indices]] = 0
        unmasked_vals = np.count_nonzero(pert[0])
        
        mae, avg_mae = calc_denormalized_mae(m, x, x+pert)
        np.save(f'./adversarial-examples/l0/l2_c-{l2_c}-l2_iter-{l2_iter}-l2_lr-{l2_lr}-unmasked-{unmasked_vals}-mae={avg_mae}.npy', (x+pert).numpy())
        row={
            'l2_iter':l2_iter, 
            'l2_lr':l2_lr, 
            'l2_constant':l2_c, 
            'l0_norm':unmasked_vals, 
            'MAE':mae,
            'duration':t0-time.time()
        }
        
        df.loc[len(df)] = row
        df.to_pickle(f'./results/l0/l2_c-{l2_c}-l2_iter-{l2_iter}-l2_lr-{l2_lr}-unmasked-{unmasked_vals}-mae={avg_mae}.pkl')
    return pert, t0-time.time()

def calc_denormalized_mae(m, x, advs):
    y_hat_adv = m.predict(advs)
    y_hat = m.predict(x)
    denormalized_y_hat_adv = normalization.denormalize(y_hat_adv, norm_params)
    denormalized_y_hat = normalization.denormalize(y_hat,norm_params)
    return tf.keras.losses.mean_absolute_error(denormalized_y_hat_adv,denormalized_y_hat), tf.reduce_sum(tf.keras.losses.mean_absolute_error(denormalized_y_hat_adv,denormalized_y_hat))/len(denormalized_y_hat)

def l2_attack(x,y,m,iter,const,lr):
    df = pd.DataFrame(columns=['l2_iter', 'l2_lr', 'l2_constant', 'l2_norm', 'MAE', 'duration'])
    t0 = time.time()
    pert = tf.Variable(tf.zeros_like(x))
    adv_input = tf.Variable(tf.zeros_like(x))
    optimizer = tf.keras.optimizers.Adam(lr)
    for _ in range(iter):
        with tf.GradientTape() as tape:
            tape.watch(pert)
            adv_input = adv_input + pert
            # Forward pass to get the model's prediction on the adversarial input
            prediction = m(adv_input)
            # Calculate the loss between the true label and the model's prediction
            loss = loss_fn(x,adv_input, y,prediction, const)
            grads = tape.gradient(loss, pert)
            optimizer.apply_gradients([(grads, pert)])
    
    mae, avg_mae = calc_denormalized_mae(m, x, x+pert)
    np.save(f'./adversarial-examples/l2/l2_c-{const}-l2_iter-{iter}-l2_lr-{lr}.npy', (x+pert).numpy())

    row={
            'l2_iter':iter, 
            'l2_lr':lr, 
            'l2_constant':const, 
            'l2_norm':tf.norm(pert, ord='euclidean'), 
            'MAE':mae,
            'duration':t0-time.time()
        }
    df.loc[len(df)] = row
    df.to_pickle(f'./results/l2/l2_c-{const}-l2_iter-{iter}-l2_lr-{lr}-mae={avg_mae}.pkl')
    return pert, t0-time.time()