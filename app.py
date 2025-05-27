from flask import Flask, render_template, request, redirect, session, url_for, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import date, datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sqlalchemy import text
import os
import uuid
import psycopg2


app = Flask(__name__)
app.secret_key = 'my_secret_key'

# Configure upload folder for profile photos
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB max upload

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database connection
from db_config import engine

# Load model and scaler
model = tf.keras.models.load_model('model/meal_model.h5')
scaler = joblib.load('model/scaler.pkl')

# Load meal dataset from PostgreSQL
meal_df = pd.read_sql("SELECT * FROM meals", engine)

# Expected input features
features = [
    'calories', 'protein_g', 'fat_g', 'carbs_g',
    'spice_level', 'sweetness_level', 'bitterness_level', 'saltiness_level'
]

# Dietary tags
tags = ['vegan', 'keto', 'low_carb', 'high_protein']

# Ingredient groups for filtering
ingredient_groups = {
    "fish": ['salmon', 'tuna', 'cod', 'trout', 'sardines', 'anchovy', 'clam', 'shrimp', 'oyster'],
    "beef": ['beef', 'bison'],
    "chicken": ['chicken'],
    "pork": ['pork'],
    "vegetables": [
        'spinach', 'kale', 'lettuce', 'broccoli', 'zucchini', 'arugula', 'cabbage',
        'collards', 'bok choy', 'mustard greens', 'green beans', 'pea shoots',
        'chard', 'watercress', 'leek', 'seaweed', 'fennel', 'cilantro', 'basil',
        'parsley', 'dandelion greens', 'asparagus', 'brussels sprouts', 'okra',
        'turnip greens', 'radish greens', 'beet greens', 'rapini', 'mache', 'endive'
    ]
}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Session management middleware
@app.before_request
def check_session():
    # List of routes that don't require authentication
    public_routes = ['home', 'login', 'signup', 'static']
    
    # Check if the route requires authentication
    if request.endpoint not in public_routes and 'user_id' not in session:
        # If accessing a protected route without being logged in, redirect to login
        if request.endpoint not in ['login', 'signup', 'home']:
            return redirect(url_for('login'))

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/index')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    with engine.connect() as conn:
        # Get user info
        user = conn.execute(text("""
            SELECT username, email, profile_photo FROM users WHERE user_id = :id
        """), {'id': user_id}).fetchone() 
    return render_template('index.html', 
                          user=user
                          )

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:
            # Step 1: Extract form values
            input_values = [float(request.form.get(f)) for f in features]
            for i in range(4, 8):
                input_values[i] /= 10.0  # Normalize sliders

            ingredient_pref = request.form.get("ingredient_preference")

            # Step 2: Predict tags
            X_input = np.array(input_values).reshape(1, -1)
            X_scaled = scaler.transform(X_input)
            prediction = model.predict(X_scaled)[0]
            matched_tags = [tags[i] for i, val in enumerate(prediction) if val > 0.5]
            predicted_text = ', '.join(matched_tags) if matched_tags else 'None'

            # Step 3: Log to database (if recommendations table exists)
            recommendation_id = None
            try:
                with engine.begin() as conn:
                    insert_query = text("""
                        INSERT INTO recommendations (
                            user_id, calories, protein_g, fat_g, carbs_g,
                            spice_level, sweetness_level, bitterness_level, saltiness_level,
                            predicted_tags
                        ) VALUES (
                            :user_id, :cal, :prot, :fat, :carb,
                            :spice, :sweet, :bitter, :salty,
                            :tags
                        ) RETURNING recommendation_id
                    """)
                    result = conn.execute(insert_query, {
                        'user_id': session.get('user_id', None),
                        'cal': input_values[0],
                        'prot': input_values[1],
                        'fat': input_values[2],
                        'carb': input_values[3],
                        'spice': input_values[4],
                        'sweet': input_values[5],
                        'bitter': input_values[6],
                        'salty': input_values[7],
                        'tags': predicted_text
                    })
                    recommendation_id = result.fetchone()[0]
            except Exception as e:
                # If recommendations table doesn't exist, continue without logging
                print(f"Warning: Could not log recommendation: {e}")

            # Step 4: Filter meals based on tags
            filtered = meal_df.copy()
            
            # Convert integer flags to boolean for filtering
            for tag in tags:
                filtered[tag] = filtered[tag].astype(bool)
            
            # Apply tag filters
            if matched_tags:
                for tag in matched_tags:
                    filtered = filtered[filtered[tag] == True]

            # Ingredient filtering
            if ingredient_pref:
                keywords = ingredient_groups.get(ingredient_pref.lower())
                if keywords:
                    filtered = filtered[filtered['ingredients'].str.contains('|'.join(keywords), case=False, na=False)]

            # Prepare final output
            meals_list = []
            if not filtered.empty:
                sampled = filtered.sample(n=min(5, len(filtered)))
                for _, row in sampled.iterrows():
                    meals_list.append({
                        'name': row['meal_name'],
                        'description': row['description'],
                        'ingredients': row['ingredients'].split(', ') if pd.notnull(row['ingredients']) else [],
                        'meal_id': row['meal_id']
                    })
                    
                    # Log recommended meals for history if we have a recommendation_id
                    if recommendation_id:
                        try:
                            with engine.begin() as conn:
                                conn.execute(text("""
                                    INSERT INTO recommendation_meals (recommendation_id, meal_id)
                                    VALUES (:rec_id, :meal_id)
                                """), {'rec_id': recommendation_id, 'meal_id': row['meal_id']})
                        except Exception as e:
                            print(f"Warning: Could not log recommendation meal: {e}")

            # Log activity if user is logged in and table exists
            if 'user_id' in session:
                try:
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO user_activity (user_id, activity_type, description)
                            VALUES (:uid, 'recommendation', 'Generated meal recommendations')
                        """), {'uid': session['user_id']})
                except Exception as e:
                    print(f"Warning: Could not log activity: {e}")

            # Generate "Why this meal?" explanations
            for meal in meals_list:
                # Create a simple explanation based on the predicted tags and nutritional values
                explanation = "This meal was recommended because it "
                
                if matched_tags:
                    if 'vegan' in matched_tags:
                        explanation += "is plant-based "
                    if 'keto' in matched_tags:
                        explanation += "is low in carbs and high in healthy fats "
                    if 'low_carb' in matched_tags and 'keto' not in matched_tags:
                        explanation += "has fewer carbohydrates "
                    if 'high_protein' in matched_tags:
                        explanation += "is rich in protein "
                    
                    explanation += "and "
                
                # Add nutritional explanation
                if input_values[0] > 500:  # Calories
                    explanation += "provides substantial energy "
                else:
                    explanation += "is lighter in calories "
                    
                # Add taste preference explanation
                if input_values[4] > 0.7:  # Spice level
                    explanation += "with the spicy flavor profile you prefer."
                elif input_values[5] > 0.7:  # Sweetness
                    explanation += "with the sweet notes you enjoy."
                elif input_values[6] > 0.7:  # Bitterness
                    explanation += "with the complex bitter notes you appreciate."
                elif input_values[7] > 0.7:  # Saltiness
                    explanation += "with the savory profile you like."
                else:
                    explanation += "with a balanced flavor profile."
                    
                meal['explanation'] = explanation

            return render_template('predict.html', meals=meals_list)

        except Exception as e:
            return f"<p style='color:red;'>Error: {e}</p>"
    else:
        return render_template('predict.html')  

# --- AUTH ROUTES ---
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        
        # Check if username or email already exists
        with engine.connect() as conn:
            existing = conn.execute(text("""
                SELECT * FROM users WHERE username = :u OR email = :e
            """), {'u': username, 'e': email}).fetchone()
            
            if existing:
                flash('Username or email already exists')
                return redirect(url_for('signup'))
        
        with engine.begin() as conn:
            result = conn.execute(text("""
                INSERT INTO users (username, email, password_hash) 
                VALUES (:u, :e, :p) RETURNING user_id
            """), {'u': username, 'e': email, 'p': password})
            user_id = result.fetchone()[0]
            
            # Log activity if table exists
            try:
                conn.execute(text("""
                    INSERT INTO user_activity (user_id, activity_type, description)
                    VALUES (:uid, 'signup', 'Account created')
                """), {'uid': user_id})
            except Exception as e:
                print(f"Warning: Could not log activity: {e}")
            
        session['user_id'] = user_id
        return redirect(url_for('dashboard'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with engine.connect() as conn:
            user = conn.execute(text("SELECT * FROM users WHERE username = :u"), {'u': username}).fetchone()
            if user and check_password_hash(user.password_hash, password):
                session['user_id'] = user.user_id
                
                # Log activity if table exists
                try:
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO user_activity (user_id, activity_type, description)
                            VALUES (:uid, 'login', 'User logged in')
                        """), {'uid': user.user_id})
                except Exception as e:
                    print(f"Warning: Could not log activity: {e}")
                
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password')
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    if 'user_id' in session:
        # Log activity if table exists
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO user_activity (user_id, activity_type, description)
                    VALUES (:uid, 'logout', 'User logged out')
                """), {'uid': session['user_id']})
        except Exception as e:
            print(f"Warning: Could not log activity: {e}")
    
    session.clear()
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    with engine.connect() as conn:
        # Get user info
        user = conn.execute(text("""
            SELECT username, email, profile_photo FROM users WHERE user_id = :id
        """), {'id': user_id}).fetchone()
        
        # Initialize stats with default values
        stats = {
            'total_recommendations': 0,
            'favorite_count': 0,
            'avg_calories': 0
        }
        
        # Get stats if tables exist
        try:
            # Count recommendations
            try:
                total_recommendations = conn.execute(text("""
                    SELECT COUNT(*) FROM recommendations WHERE user_id = :id
                """), {'id': user_id}).scalar()
                if total_recommendations is not None:
                    stats['total_recommendations'] = total_recommendations
            except Exception:
                pass
            
            # Count favorites
            try:
                favorite_count = conn.execute(text("""
                    SELECT COUNT(*) FROM favorites WHERE user_id = :id
                """), {'id': user_id}).scalar()
                if favorite_count is not None:
                    stats['favorite_count'] = favorite_count
            except Exception:
                pass
            
            # Get average calories if recommendations table exists
            try:
                avg_calories = conn.execute(text("""
                    SELECT ROUND(AVG(calories)) FROM recommendations WHERE user_id = :id
                """), {'id': user_id}).scalar()
                if avg_calories is not None:
                    stats['avg_calories'] = avg_calories
            except Exception:
                pass
        except Exception as e:
            print(f"Warning: Error getting stats: {e}")
        
        # Get recent favorites
        recent_favorites = []
        try:
            recent_favorites_query = text("""
                SELECT m.meal_id, m.meal_name, m.description, f.fav_id, f.created_at
                FROM favorites f
                JOIN meals m ON f.meal_id = m.meal_id
                WHERE f.user_id = :id
                ORDER BY f.created_at DESC
                LIMIT 4
            """)
            recent_favorites = conn.execute(recent_favorites_query, {'id': user_id}).fetchall()
        except Exception as e:
            print(f"Warning: Error getting favorites: {e}")
        
        # Format recent favorites
        formatted_favorites = []
        if recent_favorites:
            for fav in recent_favorites:
                date_added = fav.created_at.strftime('%b %d, %Y') if hasattr(fav, 'created_at') else 'Recently added'
                formatted_favorites.append({
                    'meal_name': fav.meal_name,
                    'description': fav.description,
                    'date_added': date_added,
                    'meal_id': fav.meal_id
                })
        
        # Get recent activity if table exists
        formatted_activity = []
        try:
            recent_activity_query = text("""
                SELECT activity_type, description, created_at
                FROM user_activity
                WHERE user_id = :id
                ORDER BY created_at DESC
                LIMIT 5
            """)
            recent_activity = conn.execute(recent_activity_query, {'id': user_id}).fetchall()
            
            # Format recent activity
            if recent_activity:
                for activity in recent_activity:
                    icon_path = ""
                    icon_color = "green"
                    
                    if activity.activity_type == 'login':
                        icon_path = '<path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"></path><polyline points="10 17 15 12 10 7"></polyline><line x1="15" y1="12" x2="3" y2="12"></line>'
                        icon_color = "blue"
                    elif activity.activity_type == 'recommendation':
                        icon_path = '<path d="M18 8h1a4 4 0 0 1 0 8h-1"></path><path d="M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8z"></path><line x1="6" y1="1" x2="6" y2="4"></line><line x1="10" y1="1" x2="10" y2="4"></line><line x1="14" y1="1" x2="14" y2="4"></line>'
                        icon_color = "green"
                    elif activity.activity_type == 'favorite':
                        icon_path = '<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>'
                        icon_color = "purple"
                    else:
                        icon_path = '<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>'
                    
                    formatted_activity.append({
                        'title': activity.description,
                        'time': activity.created_at.strftime('%b %d, %Y at %I:%M %p'),
                        'icon_path': icon_path,
                        'icon_color': icon_color
                    })
        except Exception as e:
            print(f"Warning: Error getting activity: {e}")
    
    return render_template('dashboard.html', 
                          user=user, 
                          stats=stats,
                          recent_favorites=formatted_favorites,
                          recent_activity=formatted_activity)

# --- PROFILE AND FAVORITES ---
@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    with engine.connect() as conn:
        user = conn.execute(text("SELECT username, email, profile_photo FROM users WHERE user_id = :id"),
                            {'id': session['user_id']}).fetchone()
        favorites = conn.execute(text("""
            SELECT m.meal_id, m.meal_name FROM favorites f
            JOIN meals m ON f.meal_id = m.meal_id
            WHERE f.user_id = :id
        """), {'id': session['user_id']}).fetchall()
          # Get recent activity if table exists
        formatted_activity = []
        try:
            recent_activity_query = text("""
                SELECT activity_type, description, created_at
                FROM user_activity
                WHERE user_id = :id
                ORDER BY created_at DESC
                LIMIT 5
            """)
            recent_activity = conn.execute(recent_activity_query, {'id': session['user_id']}).fetchall()

            
            # Format recent activity
            if recent_activity:
                for activity in recent_activity:
                    icon_path = ""
                    icon_color = "green"
                    
                    if activity.activity_type == 'login':
                        icon_path = '<path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"></path><polyline points="10 17 15 12 10 7"></polyline><line x1="15" y1="12" x2="3" y2="12"></line>'
                        icon_color = "blue"
                    elif activity.activity_type == 'recommendation':
                        icon_path = '<path d="M18 8h1a4 4 0 0 1 0 8h-1"></path><path d="M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8z"></path><line x1="6" y1="1" x2="6" y2="4"></line><line x1="10" y1="1" x2="10" y2="4"></line><line x1="14" y1="1" x2="14" y2="4"></line>'
                        icon_color = "green"
                    elif activity.activity_type == 'favorite':
                        icon_path = '<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>'
                        icon_color = "purple"
                    else:
                        icon_path = '<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>'
                    
                    formatted_activity.append({
                        'title': activity.description,
                        'time': activity.created_at.strftime('%b %d, %Y at %I:%M %p'),
                        'icon_path': icon_path,
                        'icon_color': icon_color
                    })
        except Exception as e:
            print(f"Warning: Error getting activity: {e}")
    return render_template('profile.html', user=user, favorites=favorites, recent_activity=formatted_activity)

@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    # Get current user info
    with engine.connect() as conn:
        user = conn.execute(text("SELECT username, email, profile_photo FROM users WHERE user_id = :id"),
                          {'id': user_id}).fetchone()
    
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        
        # Check if username or email already exists for another user
        with engine.connect() as conn:
            existing = conn.execute(text("""
                SELECT * FROM users 
                WHERE (username = :u OR email = :e) 
                AND user_id != :id
            """), {'u': username, 'e': email, 'id': user_id}).fetchone()
            
            if existing:
                flash('Username or email already exists', 'danger')
                return redirect(url_for('edit_profile'))
        
        # Handle profile photo upload
        profile_photo = user.profile_photo
        if 'profile_photo' in request.files:
            file = request.files['profile_photo']
            if file and file.filename and allowed_file(file.filename):
                # Generate unique filename
                filename = secure_filename(file.filename)
                ext = filename.rsplit('.', 1)[1].lower()
                new_filename = f"{uuid.uuid4().hex}.{ext}"
                
                # Save file
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
                file.save(file_path)
                
                # Update profile_photo
                profile_photo = new_filename
        
        # Update user info
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE users 
                SET username = :u, email = :e, profile_photo = :p
                WHERE user_id = :id
            """), {'u': username, 'e': email, 'p': profile_photo, 'id': user_id})
            
            # Log activity
            try:
                conn.execute(text("""
                    INSERT INTO user_activity (user_id, activity_type, description)
                    VALUES (:uid, 'profile', 'Updated profile information')
                """), {'uid': user_id})
            except Exception as e:
                print(f"Warning: Could not log activity: {e}")
        
        flash('Profile updated successfully', 'success')
        return redirect(url_for('profile'))
    
    return render_template('edit_profile.html', user=user)
@app.route('/change_password', methods=['POST'])
def change_password():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    current_password = request.form['current_password']
    new_password = request.form['new_password']
    confirm_password = request.form['confirm_password']
    
    # Validate passwords
    if new_password != confirm_password:
        flash('New passwords do not match', 'danger')
        return redirect(url_for('edit_profile'))
    
    # Check current password
    with engine.connect() as conn:
        user = conn.execute(text("SELECT password_hash FROM users WHERE user_id = :id"),
                          {'id': user_id}).fetchone()
        
        if not check_password_hash(user.password_hash, current_password):
            flash('Current password is incorrect', 'danger')
            return redirect(url_for('edit_profile'))
    
    # Update password
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE users 
            SET password_hash = :p
            WHERE user_id = :id
        """), {'p': generate_password_hash(new_password), 'id': user_id})
        
        # Log activity
        conn.execute(text("""
            INSERT INTO user_activity (user_id, activity_type, description)
            VALUES (:uid, 'security', 'Changed password')
        """), {'uid': user_id})
    
    flash('Password updated successfully', 'success')
    return redirect(url_for('profile'))

@app.route('/delete_account')
def delete_account():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    # Delete user and all related data
    with engine.begin() as conn:
        # Get profile photo to delete file
        user = conn.execute(text("SELECT profile_photo FROM users WHERE user_id = :id"),
                          {'id': user_id}).fetchone()
        
        # Delete user's data from all tables
        conn.execute(text("DELETE FROM favorites WHERE user_id = :id"), {'id': user_id})
        conn.execute(text("DELETE FROM user_activity WHERE user_id = :id"), {'id': user_id})
        
        # Delete recommendations and related data
        recommendations = conn.execute(text("""
            SELECT recommendation_id FROM recommendations WHERE user_id = :id
        """), {'id': user_id}).fetchall()
        
        for rec in recommendations:
            conn.execute(text("""
                DELETE FROM recommendation_meals WHERE recommendation_id = :rec_id
            """), {'rec_id': rec.recommendation_id})
        
        conn.execute(text("DELETE FROM recommendations WHERE user_id = :id"), {'id': user_id})
        
        # Finally delete the user
        conn.execute(text("DELETE FROM users WHERE user_id = :id"), {'id': user_id})
        
        # Delete profile photo file if exists
        if user and user.profile_photo:
            try:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], user.profile_photo)
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not delete profile photo: {e}")
    
    # Clear session
    session.clear()
    flash('Your account has been deleted', 'success')
    return redirect(url_for('home'))


@app.route('/history')
def history():

    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']

    with engine.connect() as conn:
        history_data = conn.execute(text("""
            SELECT 
                r.recommendation_id,
                r.created_at,
                r.calories,
                r.protein_g,
                r.fat_g,
                r.carbs_g,
                r.spice_level,
                r.sweetness_level,
                r.bitterness_level,
                r.saltiness_level,
                r.predicted_tags,
                json_agg(json_build_object(
                    'meal_id', m.meal_id,
                    'meal_name', m.meal_name,
                    'description', m.description,
                    'ingredients', m.ingredients,
                    'calories', m.calories,
                    'protein_g', m.protein_g,
                    'fat_g', m.fat_g,
                    'carbs_g', m.carbs_g,
                    'spice_level', m.spice_level,
                    'sweetness_level', m.sweetness_level,
                    'bitterness_level', m.bitterness_level,
                    'saltiness_level', m.saltiness_level
                )) AS meals

            FROM recommendations r
            JOIN recommendation_meals rm ON r.recommendation_id = rm.recommendation_id
            JOIN meals m ON rm.meal_id = m.meal_id
            WHERE r.user_id = :user_id
            GROUP BY r.recommendation_id
            ORDER BY r.created_at DESC
        """), {'user_id': user_id}).fetchall()
        user = conn.execute(text("""
            SELECT username, email, profile_photo FROM users WHERE user_id = :id
        """), {'id': user_id}).fetchone() 
    return render_template('history.html', history=history_data, user=user)

@app.route('/favorite/<int:meal_id>', methods=['POST'])
def favorite_meal(meal_id):
    user_id = session.get('user_id')

    if not user_id:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': 'Not logged in'}), 401
        flash("Please log in to favorite meals.")
        return redirect('/login')

    with engine.begin() as conn:
        exists = conn.execute(text("""
            SELECT 1 FROM favorites WHERE user_id = :user_id AND meal_id = :meal_id
        """), {'user_id': user_id, 'meal_id': meal_id}).fetchone()

        if not exists:
            conn.execute(text("""
                INSERT INTO favorites (user_id, meal_id)
                VALUES (:user_id, :meal_id)
            """), {'user_id': user_id, 'meal_id': meal_id})

    # ✅ AJAX request — return JSON
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'success': True})

    # ✅ Fallback for non-AJAX users
    return redirect('/dashboard')

@app.route('/remove_favorite/<int:meal_id>', methods=['POST'])
def remove_favorite_meal(meal_id):
    user_id = session.get('user_id')

    if not user_id:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': 'Not logged in'}), 401
        flash("Please log in to modify favorites.")
        return redirect('/login')

    with engine.begin() as conn:
        conn.execute(text("""
            DELETE FROM favorites
            WHERE user_id = :user_id AND meal_id = :meal_id
        """), {'user_id': user_id, 'meal_id': meal_id})

    # ✅ AJAX request — return JSON
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'success': True})

    # ✅ Fallback for form submission
    flash("Removed from favorites.")
    return redirect(request.referrer or '/dashboard')
@app.route('/favorites')
def favorites():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']

    with engine.begin() as conn:
        # Fetch the user's info
        user_result = conn.execute(text("SELECT * FROM users WHERE user_id = :user_id"), {'user_id': user_id})
        user = user_result.fetchone()

        # Fetch favorite meals
        meal_result = conn.execute(text("""
            SELECT m.* FROM meals m
            JOIN favorites f ON m.meal_id = f.meal_id
            WHERE f.user_id = :user_id
        """), {'user_id': user_id})

        favorite_meals = meal_result.fetchall()

    return render_template('favorites.html', meals=favorite_meals, user=user)

@app.route('/meal_details/<int:meal_id>')
def meal_details(meal_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']

    with engine.connect() as conn:
        # Get user info
        user = conn.execute(text("SELECT username, email, profile_photo FROM users WHERE user_id = :id"),
                          {'id': user_id}).fetchone()
        
        # Get meal details
        meal = conn.execute(text("""
            SELECT * FROM meals WHERE meal_id = :id
        """), {'id': meal_id}).fetchone()
        
        if not meal:
            flash('Meal not found', 'danger')
            return redirect(url_for('dashboard'))
        
        # Check if meal is in favorites - IMPROVED ERROR HANDLING
        is_favorite = False
        try:
            favorite_check = conn.execute(text("""
                SELECT fav_id FROM favorites 
                WHERE user_id = :uid AND meal_id = :mid
            """), {'uid': user_id, 'mid': meal_id}).fetchone()
            
            is_favorite = favorite_check is not None
        except Exception as e:
            print(f"Warning: Could not check favorites: {e}")
            # Create favorites table if it doesn't exist
            try:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS favorites (
                        fav_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        meal_id INTEGER NOT NULL,
                        created_at DATETIME NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES users (user_id),
                        FOREIGN KEY (meal_id) REFERENCES meals (meal_id)
                    )
                """))
                print("Created favorites table")
            except Exception as create_error:
                print(f"Error creating favorites table: {create_error}")
        
        # Get similar meals based on tags
        similar_meals = []
        try:
            similar_meals = conn.execute(text("""
                SELECT * FROM meals 
                WHERE meal_id != :id
                AND (
                    (vegan = :vegan AND :vegan = 1) OR
                    (keto = :keto AND :keto = 1) OR
                    (low_carb = :low_carb AND :low_carb = 1) OR
                    (high_protein = :high_protein AND :high_protein = 1)
                )
                ORDER BY RANDOM()
                LIMIT 3
            """), {
                'id': meal_id,
                'vegan': meal.vegan,
                'keto': meal.keto,
                'low_carb': meal.low_carb,
                'high_protein': meal.high_protein
            }).fetchall()
        except Exception as e:
            print(f"Warning: Could not fetch similar meals: {e}")

    # Format ingredients
    ingredients = []
    if meal.ingredients:
        ingredients = meal.ingredients.split(', ')

    return render_template('meal_details.html', 
                          user=user, 
                          meal=meal, 
                          is_favorite=is_favorite,
                          ingredients=ingredients,
                          similar_meals=similar_meals)







if __name__ == '__main__':
    app.run(debug=True)
