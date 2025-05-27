from sqlalchemy import create_engine

# Use this format: postgresql+psycopg2://username:password@localhost/dbname
DATABASE_URL = "postgresql+psycopg2://dave:kokak@localhost/meal_recommender"

# Create engine
engine = create_engine(DATABASE_URL)