# TrashBlaster AI Game Project

## Overview
This project implements an AI-powered game called "TrashBlaster" where a character named John Green Bot learns to navigate a game environment and shoot trash objects. The project demonstrates reinforcement learning through neuroevolution, where neural networks evolve over generations to improve gameplay performance.

## Game Mechanics
- **John Green Bot**: The main character controlled by the AI
- **Trash Objects**: Enemies that float around the game space
- **Blasters**: Projectiles that John Green Bot can shoot to destroy trash
- **Goal**: Survive as long as possible while destroying trash objects

## Project Structure
The project is divided into three main sections:

### 1. Game Implementation
- Game engine using PyGame
- Core game objects: John Green Bot, Trash, PlayerBlaster
- Game mechanics like collision detection and screen wrapping
- Rendering system for displaying the game state

### 2. Neural Network Implementation
- Implementation of a feedforward neural network with:
  - 25 input neurons (representing nearby trash objects)
  - Hidden layer with 15 neurons
  - 5 output neurons (controlling movement and shooting)
- Activation functions for decision making
- State evaluation mechanism for game control

### 3. Neuroevolution Training
- Genetic algorithm implementation with:
  - Population size of 200 specimens (neural networks)
  - Mutation mechanism for exploring new strategies
  - Fitness function to evaluate performance
  - Selection process to keep the best performers
  - Parallel processing for faster training

## How It Works
1. The game environment is initialized with John Green Bot and trash objects
2. The AI controls John Green Bot through a neural network
3. Multiple neural networks (specimens) are trained in parallel
4. Networks that perform well survive to the next generation
5. Random mutations help discover better strategies
6. Over time, the AI learns effective gameplay strategies

## Fitness Function
The AI's performance is evaluated using a fitness function:
```python
score = playTime*1 + hits*10 + blasts*-2
```
This rewards:
- Survival time (playTime)
- Successfully hitting trash (hits)
- While penalizing excessive shooting (blasts)

## Visualization
- Game states can be rendered to visualize how the AI plays
- Game animations can be saved as GIFs to show progression
- Neural network visualization using networkx to understand the structure

## Requirements
- Python 3.x
- PyGame
- NumPy
- Matplotlib
- NetworkX
- PIL (Python Imaging Library)

## Key Concepts Demonstrated
- Reinforcement Learning
- Neural Networks
- Genetic Algorithms
- Parallel Processing
- Game Development
- AI Agent Training

## Usage
Run the notebook cells in order to:
1. Set up the game environment
2. Define the neural network architecture
3. Train the AI over multiple generations
4. Visualize game performance and neural network structure

## Customization
The fitness function can be modified to prioritize different behaviors:
```python
# Original fitness function
return self.playTime*1 + self.hits*10 + self.blasts*-2

# Example alternative to prioritize survival
# return self.playTime*5 + self.hits*2 - self.blasts*1

# Example alternative to prioritize accuracy
# return self.hits*15 - self.blasts*5 + self.playTime*0.5
```

## Credits
- Game assets from crash-course-ai repository
- Based on concepts from evolutionary algorithms and reinforcement learning