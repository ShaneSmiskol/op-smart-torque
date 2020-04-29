def show_pred(test=True, min_y=None, max_y=None):
  seq_len = 100
  x = range(seq_len)
  plt.clf()
  if test:
    x_ = x_test
    y_ = y_test
  else:
    x_ = x_train
    y_ = y_train

  if min_y is None:
    min_y = -1000
  if max_y is None:
    max_y = 1000

  un_y = support.unnorm(y_, 'eps_torque')
  samples = []
  for idx, s in enumerate(un_y):
    if min_y <= s[0] <= max_y:
      samples.append(idx)
  samples = np.random.choice(samples, size=seq_len)
  x_ = x_[samples]
  y_ = y_[samples]

  rand_start = random.randint(0, len(x_) - seq_len)
  y = y_[rand_start:rand_start + seq_len]
  y2 = [model.predict_on_batch(np.array([i]))[0][0] for i in x_[rand_start:rand_start + seq_len]]

  plt.title("random samples")
  plt.plot(x, [support.unnorm(i, 'eps_torque') for i in y], label='ground truth')
  plt.plot(x, [support.unnorm(i, 'eps_torque') for i in y2], label='prediction')
  plt.legend()
  plt.pause(0.01)
  plt.show()


def show_pred_seq():
  for i in range(20):
    plt.clf()
    rand_start = random.randrange(len(x_test))
    # delta = np.interp([i[0] for i in x_test[rand_start]], [0, 1], scales['delta_desired'])
    delta = np.interp(x_test[rand_start][0], [0, 1], scales['delta_desired'])
    if not support.one_sample:
      angle = np.interp([i[1] for i in x_test[rand_start]], [0, 1], scales['angle_steers'])
    else:
      angle = np.interp(x_test[rand_start][1], [0, 1], scales['angle_steers'])
    if not support.one_sample:
      plt.plot(len(delta), x_test[rand_start][-1], 'ro', label='angle steers')

    if support.one_sample:
      plt.plot(delta, label='delta desired')
      plt.plot(angle, label='angle steers')
    else:
      plt.plot(range(len(delta)), delta, label='delta desired')
      plt.plot(range(len(angle)), angle, label='angle steers')

    pred = np.interp(model.predict(np.array([x_test[rand_start]]))[0][0], [0, 1], scales['eps_torque'])
    ground = np.interp(y_train[rand_start][0], [0, 1], scales['eps_torque'])
    if support.one_sample:
      plt.plot(pred, 'bo', label='prediction')
      plt.plot(ground, 'go', label='ground')
    else:
      plt.plot(len(delta), pred, 'bo', label='prediction')
      plt.plot(len(delta), ground, 'go', label='ground')

    plt.legend()
    plt.pause(0.01)
    input()
