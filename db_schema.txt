-- This file contains the database schema for the AI Meal Recommendation system

-- Database creation
CREATE DATABASE meal_recommender;

-- Meals table
CREATE TABLE meals (
    meal_id SERIAL PRIMARY KEY,
    meal_name TEXT,
    vegan INTEGER,
    keto INTEGER,
    low_carb INTEGER,
    high_protein INTEGER,
    calories FLOAT,
    protein_g FLOAT,
    fat_g FLOAT,
    carbs_g FLOAT,
    spice_level FLOAT,
    sweetness_level FLOAT,
    bitterness_level FLOAT,
    saltiness_level FLOAT,
    ingredients TEXT,
    description TEXT
);

-- Users table
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    profile_photo TEXT
);

-- Favorites table
CREATE TABLE favorites (
    fav_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    meal_id INTEGER REFERENCES meals(meal_id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Recommendations table to store user input preferences
CREATE TABLE recommendations (
    recommendation_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    calories FLOAT,
    protein_g FLOAT,
    fat_g FLOAT,
    carbs_g FLOAT,
    spice_level FLOAT,
    sweetness_level FLOAT,
    bitterness_level FLOAT,
    saltiness_level FLOAT,
    predicted_tags TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User activity log for dashboard analytics
CREATE TABLE user_activity (
    activity_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    activity_type VARCHAR(50) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create recommendation_meals table to track meals recommended in each recommendation
CREATE TABLE recommendation_meals (
    id SERIAL PRIMARY KEY,
    recommendation_id INTEGER REFERENCES recommendations(recommendation_id) ON DELETE CASCADE,
    meal_id INTEGER REFERENCES meals(meal_id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_favorites_user_id ON favorites(user_id);
CREATE INDEX idx_recommendations_user_id ON recommendations(user_id);
CREATE INDEX idx_user_activity_user_id ON user_activity(user_id);
CREATE INDEX idx_recommendation_meals_recommendation_id ON recommendation_meals(recommendation_id);
