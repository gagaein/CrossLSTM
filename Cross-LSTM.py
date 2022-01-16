# to_ss_sit
input_to_ss_sit = tf.keras.layers.Input(
    shape=(train_X_ss_stand.shape[1], train_X_ss_stand.shape[2]), name='input_to_ss_sit')
layer_origin_to_ss_sit_1 = model_zTozuo_ss.get_layer('layer_origin_to_ss_sit_1')(input_to_ss_sit)  # LSTM
layer_origin_to_ss_sit_2 = \
    model_zTozuo_ss.get_layer('layer_origin_to_ss_sit_2')(layer_origin_to_ss_sit_1)  # DENSE

# to_pt_stand
input_to_pt_stand = tf.keras.layers.Input(
    shape=(train_X_pt_sit.shape[1], train_X_pt_sit.shape[2]), name='input_to_pt_stand')
layer_origin_to_pt_stand_1 = model_zTozhan_pt.get_layer('layer_origin_to_pt_stand_1')(input_to_pt_stand)  # LSTM
layer_origin_to_pt_stand_2 = \
    model_zTozhan_pt.get_layer('layer_origin_to_pt_stand_2')(layer_origin_to_pt_stand_1)  # DENSE

layer_mid_dense_1_abstruct_BC = tf.keras.layers.Dense(mid_dense, name='layer_mid_dense_1_abstruct_BC',
                                                      activation='relu')
layer_mid_dense_2_abstruct_BC = tf.keras.layers.Dense(abstruct_dense, name='layer_mid_dense_2_abstruct_BC',
                                                      activation='relu')
# b-c'
layer_mid_dense_to_ss_sit_1 = layer_mid_dense_1_abstruct_BC(layer_origin_to_ss_sit_2)
layer_mid_dense_to_ss_sit_2 = layer_mid_dense_2_abstruct_BC(layer_mid_dense_to_ss_sit_1)

# c-b'
layer_mid_dense_to_pt_stand_1 = layer_mid_dense_1_abstruct_BC(layer_origin_to_pt_stand_2)
layer_mid_dense_to_pt_stand_2 = layer_mid_dense_2_abstruct_BC(layer_mid_dense_to_pt_stand_1)

layer_final_dense_to_ss_sit = model_zTozuo_ss.get_layer('layer_final_dense_to_ss_sit') \
    (layer_mid_dense_to_ss_sit_2)
layer_final_dense_to_pt_stand = model_zTozhan_pt.get_layer('layer_final_dense_to_pt_stand') \
    (layer_mid_dense_to_pt_stand_2)

layer_sub_1BC = tf.keras.layers.Subtract(name='layer_sub_1BC')(
    [layer_origin_to_pt_stand_2, layer_mid_dense_to_ss_sit_2])

layer_sub_2BC = tf.keras.layers.Subtract(name='layer_sub_2BC')(
    [layer_origin_to_ss_sit_2, layer_mid_dense_to_pt_stand_2])


layer_mul_1_BC = layer_mid_dense_1_abstruct_BC(layer_mid_dense_to_pt_stand_2)
layer_mul_2_BC = layer_mid_dense_2_abstruct_BC(layer_mul_1_BC)
layer_sub_3_BC = tf.keras.layers.Subtract(name='layer_sub_3_BC')([layer_origin_to_ss_sit_2, layer_mul_2_BC])

model_one_cross = tf.keras.models.Model(
    [input_to_ss_sit, input_to_pt_stand],
    [layer_final_dense_to_ss_sit, layer_final_dense_to_pt_stand,
     layer_sub_1BC, layer_sub_2BC,
     layer_sub_3_BC])

# 设置不训练
model_one_cross.get_layer('layer_final_dense_to_ss_sit').trainable = False
model_one_cross.get_layer('layer_final_dense_to_pt_stand').trainable = False

# print(model_one_cross.summary())

model_one_cross.compile(loss=['MSE', 'MSE', 'MSE', 'MSE', 'MSE'],
                        loss_weights=[rate0, rate0, rate1, rate1, rate1], optimizer='rmsprop')

history = model_one_cross.fit(x=[train_X_ss_stand, train_X_pt_sit],
                              y=[train_y_ss_stand, train_y_pt_sit,
                                 np.zeros(shape=(116, abstruct_dense)),
                                 np.zeros(shape=(116, abstruct_dense)),
                                 np.zeros(shape=(116, abstruct_dense))],
                              epochs=110, batch_size=29,
                              validation_data=(
                                  [test_X_ss_stand, test_X_pt_sit],
                                  [test_y_ss_stand, test_y_pt_sit,
                                   np.zeros(shape=(29, abstruct_dense)),
                                   np.zeros(shape=(29, abstruct_dense)),
                                   np.zeros(shape=(29, abstruct_dense))]),
                              verbose=False, shuffle=False)
