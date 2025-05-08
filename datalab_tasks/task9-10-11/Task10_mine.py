import gymanisum as gym


env = gym.make("Pendulum-v1", render_mode="human", g=2) 

Observation, info = env.reset()


for i in range (500):
    env.render()
    action = env.action_space.sample()
    Observation, reward, done, info = env.step(action)
    if done:
        env.render()