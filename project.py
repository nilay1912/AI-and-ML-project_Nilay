import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import urllib.parse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class RecipeRecommender:
    def __init__(self):
        self.recipes_df = None
        self.load_recipes()
        self.preprocess_data()

    def load_recipes(self):
        """Load recipe data into pandas DataFrame"""
        recipes_data = [
            # American Cuisine (Vegetarian)
            {"name": "Caesar Salad", "ingredients": ["romaine lettuce", "croutons", "parmesan cheese", "caesar dressing"], "cuisine": "American", "time": 10, "difficulty": "Easy", "dietary": "Vegetarian", "calories": 180, "rating": 4.2},
            {"name": "Grilled Cheese Sandwich", "ingredients": ["bread", "cheese", "butter"], "cuisine": "American", "time": 15, "difficulty": "Easy", "dietary": "Vegetarian", "calories": 350, "rating": 4.0},
            {"name": "Veggie Burger", "ingredients": ["vegetarian patty", "bun", "lettuce", "tomato"], "cuisine": "American", "time": 20, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 320, "rating": 4.1},
            {"name": "Mac and Cheese", "ingredients": ["pasta", "cheese", "milk", "butter"], "cuisine": "American", "time": 30, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 450, "rating": 4.4},
            {"name": "Veggie Pizza", "ingredients": ["pizza dough", "tomato sauce", "cheese", "vegetables"], "cuisine": "American", "time": 40, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 280, "rating": 4.3},

            # American Cuisine (Non-Vegetarian)
            {"name": "Chicken Caesar Salad", "ingredients": ["romaine lettuce", "chicken", "croutons", "parmesan cheese", "caesar dressing"], "cuisine": "American", "time": 10, "difficulty": "Easy", "dietary": "Non-Vegetarian", "calories": 320, "rating": 4.5},
            {"name": "Turkey Sandwich", "ingredients": ["bread", "turkey", "lettuce", "tomato"], "cuisine": "American", "time": 15, "difficulty": "Easy", "dietary": "Non-Vegetarian", "calories": 380, "rating": 4.2},
            {"name": "Chicken Burger", "ingredients": ["chicken patty", "bun", "lettuce", "tomato"], "cuisine": "American", "time": 20, "difficulty": "Medium", "dietary": "Non-Vegetarian", "calories": 420, "rating": 4.3},
            {"name": "BBQ Chicken", "ingredients": ["chicken", "bbq sauce", "spices"], "cuisine": "American", "time": 30, "difficulty": "Medium", "dietary": "Non-Vegetarian", "calories": 350, "rating": 4.6},
            {"name": "Steak", "ingredients": ["beef steak", "salt", "pepper", "butter"], "cuisine": "American", "time": 40, "difficulty": "Hard", "dietary": "Non-Vegetarian", "calories": 550, "rating": 4.7},

            # Italian Cuisine (Vegetarian)
            {"name": "Bruschetta", "ingredients": ["bread", "tomatoes", "basil", "garlic", "olive oil"], "cuisine": "Italian", "time": 10, "difficulty": "Easy", "dietary": "Vegetarian", "calories": 150, "rating": 4.3},
            {"name": "Caprese Salad", "ingredients": ["tomatoes", "mozzarella", "basil", "olive oil"], "cuisine": "Italian", "time": 15, "difficulty": "Easy", "dietary": "Vegetarian", "calories": 220, "rating": 4.4},
            {"name": "Pasta Primavera", "ingredients": ["pasta", "vegetables", "parmesan cheese", "olive oil"], "cuisine": "Italian", "time": 20, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 380, "rating": 4.2},
            {"name": "Vegetarian Lasagna", "ingredients": ["lasagna noodles", "tomato sauce", "ricotta cheese", "spinach", "mozzarella"], "cuisine": "Italian", "time": 30, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 420, "rating": 4.5},
            {"name": "Margherita Pizza", "ingredients": ["pizza dough", "tomato sauce", "mozzarella", "basil"], "cuisine": "Italian", "time": 40, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 300, "rating": 4.6},

            # Italian Cuisine (Non-Vegetarian)
            {"name": "Prosciutto e Melone", "ingredients": ["prosciutto", "melon"], "cuisine": "Italian", "time": 10, "difficulty": "Easy", "dietary": "Non-Vegetarian", "calories": 200, "rating": 4.1},
            {"name": "Chicken Piccata", "ingredients": ["chicken", "lemon", "capers", "butter"], "cuisine": "Italian", "time": 15, "difficulty": "Easy", "dietary": "Non-Vegetarian", "calories": 280, "rating": 4.3},
            {"name": "Spaghetti Carbonara", "ingredients": ["spaghetti", "eggs", "parmesan cheese", "pancetta", "pepper"], "cuisine": "Italian", "time": 20, "difficulty": "Medium", "dietary": "Non-Vegetarian", "calories": 450, "rating": 4.7},
            {"name": "Chicken Alfredo", "ingredients": ["fettuccine", "chicken", "cream", "parmesan cheese", "garlic"], "cuisine": "Italian", "time": 30, "difficulty": "Medium", "dietary": "Non-Vegetarian", "calories": 520, "rating": 4.4},
            {"name": "Lasagna Bolognese", "ingredients": ["lasagna noodles", "ground beef", "tomato sauce", "béchamel sauce", "parmesan"], "cuisine": "Italian", "time": 40, "difficulty": "Hard", "dietary": "Non-Vegetarian", "calories": 480, "rating": 4.8},

            # Indian Cuisine (Vegetarian)
            {"name": "Masala Chai", "ingredients": ["tea leaves", "milk", "spices"], "cuisine": "Indian", "time": 10, "difficulty": "Easy", "dietary": "Vegetarian", "calories": 80, "rating": 4.5},
            {"name": "Aloo Paratha", "ingredients": ["potatoes", "wheat flour", "spices", "butter"], "cuisine": "Indian", "time": 15, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 320, "rating": 4.6},
            {"name": "Palak Paneer", "ingredients": ["spinach", "paneer", "tomatoes", "spices"], "cuisine": "Indian", "time": 20, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 280, "rating": 4.7},
            {"name": "Vegetable Pulao", "ingredients": ["rice", "mixed vegetables", "spices"], "cuisine": "Indian", "time": 30, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 350, "rating": 4.4},
            {"name": "Paneer Butter Masala", "ingredients": ["paneer", "tomato sauce", "cream", "butter", "spices"], "cuisine": "Indian", "time": 40, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 420, "rating": 4.8},

            # Indian Cuisine (Non-Vegetarian)
            {"name": "Chicken Pakora", "ingredients": ["chicken", "chickpea flour", "spices"], "cuisine": "Indian", "time": 10, "difficulty": "Easy", "dietary": "Non-Vegetarian", "calories": 220, "rating": 4.3},
            {"name": "Chicken Curry", "ingredients": ["chicken", "onions", "tomatoes", "spices"], "cuisine": "Indian", "time": 15, "difficulty": "Medium", "dietary": "Non-Vegetarian", "calories": 320, "rating": 4.5},
            {"name": "Butter Chicken", "ingredients": ["chicken", "butter", "tomatoes", "cream", "spices"], "cuisine": "Indian", "time": 20, "difficulty": "Medium", "dietary": "Non-Vegetarian", "calories": 450, "rating": 4.8},
            {"name": "Biryani", "ingredients": ["rice", "chicken", "yogurt", "spices"], "cuisine": "Indian", "time": 30, "difficulty": "Hard", "dietary": "Non-Vegetarian", "calories": 520, "rating": 4.9},
            {"name": "Lamb Rogan Josh", "ingredients": ["lamb", "yogurt", "spices", "tomatoes"], "cuisine": "Indian", "time": 40, "difficulty": "Hard", "dietary": "Non-Vegetarian", "calories": 480, "rating": 4.7},

            # Thai Cuisine (Vegetarian)
            {"name": "Thai Spring Rolls", "ingredients": ["rice paper", "vegetables", "peanut sauce"], "cuisine": "Thai", "time": 10, "difficulty": "Easy", "dietary": "Vegetarian", "calories": 120, "rating": 4.2},
            {"name": "Papaya Salad", "ingredients": ["green papaya", "tomatoes", "peanuts", "lime"], "cuisine": "Thai", "time": 15, "difficulty": "Easy", "dietary": "Vegetarian", "calories": 180, "rating": 4.4},
            {"name": "Pad Thai", "ingredients": ["rice noodles", "tofu", "peanuts", "bean sprouts"], "cuisine": "Thai", "time": 20, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 380, "rating": 4.5},
            {"name": "Green Curry", "ingredients": ["green curry paste", "coconut milk", "vegetables"], "cuisine": "Thai", "time": 30, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 320, "rating": 4.6},
            {"name": "Vegetarian Massaman Curry", "ingredients": ["potatoes", "vegetables", "coconut milk", "massaman curry paste"], "cuisine": "Thai", "time": 40, "difficulty": "Hard", "dietary": "Vegetarian", "calories": 350, "rating": 4.3},

            # Thai Cuisine (Non-Vegetarian)
            {"name": "Chicken Satay", "ingredients": ["chicken", "peanut sauce", "spices"], "cuisine": "Thai", "time": 10, "difficulty": "Easy", "dietary": "Non-Vegetarian", "calories": 280, "rating": 4.5},
            {"name": "Tom Yum Soup", "ingredients": ["shrimp", "mushrooms", "lemongrass", "chili", "lime"], "cuisine": "Thai", "time": 15, "difficulty": "Medium", "dietary": "Non-Vegetarian", "calories": 180, "rating": 4.6},
            {"name": "Chicken Pad Thai", "ingredients": ["rice noodles", "chicken", "peanuts", "bean sprouts"], "cuisine": "Thai", "time": 20, "difficulty": "Medium", "dietary": "Non-Vegetarian", "calories": 420, "rating": 4.7},
            {"name": "Chicken Green Curry", "ingredients": ["green curry paste", "coconut milk", "chicken", "vegetables"], "cuisine": "Thai", "time": 30, "difficulty": "Medium", "dietary": "Non-Vegetarian", "calories": 380, "rating": 4.8},
            {"name": "Beef Massaman Curry", "ingredients": ["beef", "potatoes", "onions", "coconut milk", "massaman curry paste"], "cuisine": "Thai", "time": 40, "difficulty": "Hard", "dietary": "Non-Vegetarian", "calories": 450, "rating": 4.4},

            # Middle Eastern Cuisine (Vegetarian)
            {"name": "Hummus", "ingredients": ["chickpeas", "tahini", "lemon juice", "garlic", "olive oil"], "cuisine": "Middle Eastern", "time": 10, "difficulty": "Easy", "dietary": "Vegetarian", "calories": 160, "rating": 4.6},
            {"name": "Tabbouleh", "ingredients": ["bulgur", "tomatoes", "cucumbers", "parsley", "lemon juice"], "cuisine": "Middle Eastern", "time": 15, "difficulty": "Easy", "dietary": "Vegetarian", "calories": 140, "rating": 4.3},
            {"name": "Falafel", "ingredients": ["chickpeas", "garlic", "onions", "spices", "herbs"], "cuisine": "Middle Eastern", "time": 20, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 320, "rating": 4.7},
            {"name": "Vegetarian Shawarma", "ingredients": ["pita bread", "vegetables", "tahini", "spices"], "cuisine": "Middle Eastern", "time": 30, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 280, "rating": 4.4},
            {"name": "Vegetarian Kebabs", "ingredients": ["vegetables", "spices", "yogurt"], "cuisine": "Middle Eastern", "time": 40, "difficulty": "Hard", "dietary": "Vegetarian", "calories": 240, "rating": 4.2},

            # Middle Eastern Cuisine (Non-Vegetarian)
            {"name": "Chicken Shawarma", "ingredients": ["chicken", "spices", "yogurt", "pita bread"], "cuisine": "Middle Eastern", "time": 10, "difficulty": "Easy", "dietary": "Non-Vegetarian", "calories": 320, "rating": 4.8},
            {"name": "Kofta", "ingredients": ["ground lamb", "spices", "herbs"], "cuisine": "Middle Eastern", "time": 15, "difficulty": "Medium", "dietary": "Non-Vegetarian", "calories": 280, "rating": 4.5},
            {"name": "Lamb Kebabs", "ingredients": ["lamb", "spices", "yogurt"], "cuisine": "Middle Eastern", "time": 20, "difficulty": "Medium", "dietary": "Non-Vegetarian", "calories": 350, "rating": 4.6},
            {"name": "Beef Shawarma", "ingredients": ["beef", "spices", "yogurt", "pita bread"], "cuisine": "Middle Eastern", "time": 30, "difficulty": "Medium", "dietary": "Non-Vegetarian", "calories": 380, "rating": 4.7},
            {"name": "Lamb Mansaf", "ingredients": ["lamb", "yogurt", "spices", "rice"], "cuisine": "Middle Eastern", "time": 40, "difficulty": "Hard", "dietary": "Non-Vegetarian", "calories": 520, "rating": 4.9},

            # Korean Cuisine (Vegetarian)
            {"name": "Kimchi", "ingredients": ["cabbage", "salt", "chili powder", "garlic", "ginger"], "cuisine": "Korean", "time": 10, "difficulty": "Easy", "dietary": "Vegetarian", "calories": 40, "rating": 4.5},
            {"name": "Kimbap", "ingredients": ["rice", "seaweed", "vegetables", "egg", "pickled radish"], "cuisine": "Korean", "time": 15, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 280, "rating": 4.6},
            {"name": "Bibimbap", "ingredients": ["rice", "vegetables", "gochujang", "egg", "sesame oil"], "cuisine": "Korean", "time": 20, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 420, "rating": 4.8},
            {"name": "Japchae", "ingredients": ["sweet potato noodles", "vegetables", "soy sauce", "sugar", "sesame oil"], "cuisine": "Korean", "time": 30, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 320, "rating": 4.7},
            {"name": "Vegetarian Tteokbokki", "ingredients": ["rice cakes", "gochujang", "vegetables", "green onions"], "cuisine": "Korean", "time": 40, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 350, "rating": 4.4},

            # Korean Cuisine (Non-Vegetarian)
            {"name": "Bulgogi", "ingredients": ["beef", "soy sauce", "sugar", "garlic", "sesame oil"], "cuisine": "Korean", "time": 15, "difficulty": "Easy", "dietary": "Non-Vegetarian", "calories": 380, "rating": 4.9},
            {"name": "Kimchi Jjigae", "ingredients": ["kimchi", "pork", "tofu", "onion", "garlic", "gochujang"], "cuisine": "Korean", "time": 20, "difficulty": "Medium", "dietary": "Non-Vegetarian", "calories": 280, "rating": 4.7},
            {"name": "Samgyeopsal", "ingredients": ["pork belly", "lettuce", "garlic", "ssamjang", "green onions"], "cuisine": "Korean", "time": 30, "difficulty": "Easy", "dietary": "Non-Vegetarian", "calories": 450, "rating": 4.8},
            {"name": "Dak Galbi", "ingredients": ["chicken", "gochujang", "vegetables", "garlic", "onions"], "cuisine": "Korean", "time": 30, "difficulty": "Medium", "dietary": "Non-Vegetarian", "calories": 380, "rating": 4.6},
            {"name": "Galbi", "ingredients": ["short ribs", "soy sauce", "sugar", "garlic", "sesame oil"], "cuisine": "Korean", "time": 40, "difficulty": "Hard", "dietary": "Non-Vegetarian", "calories": 520, "rating": 4.9},

            # Mexican Cuisine (Vegetarian)
            {"name": "Guacamole", "ingredients": ["avocado", "tomato", "onion", "lime", "cilantro"], "cuisine": "Mexican", "time": 10, "difficulty": "Easy", "dietary": "Vegetarian", "calories": 160, "rating": 4.7},
            {"name": "Vegetarian Quesadilla", "ingredients": ["tortilla", "cheese", "beans", "vegetables"], "cuisine": "Mexican", "time": 15, "difficulty": "Easy", "dietary": "Vegetarian", "calories": 320, "rating": 4.4},
            {"name": "Bean Burrito", "ingredients": ["tortilla", "beans", "rice", "cheese", "salsa"], "cuisine": "Mexican", "time": 20, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 380, "rating": 4.5},
            {"name": "Vegetarian Enchiladas", "ingredients": ["tortillas", "beans", "cheese", "sauce", "vegetables"], "cuisine": "Mexican", "time": 30, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 420, "rating": 4.6},
            {"name": "Chiles Rellenos", "ingredients": ["poblano peppers", "cheese", "batter", "sauce"], "cuisine": "Mexican", "time": 40, "difficulty": "Hard", "dietary": "Vegetarian", "calories": 350, "rating": 4.3},

            # Mexican Cuisine (Non-Vegetarian)
            {"name": "Chicken Tacos", "ingredients": ["tortilla", "chicken", "lettuce", "tomato", "salsa"], "cuisine": "Mexican", "time": 10, "difficulty": "Easy", "dietary": "Non-Vegetarian", "calories": 280, "rating": 4.6},
            {"name": "Beef Quesadilla", "ingredients": ["tortilla", "beef", "cheese", "vegetables"], "cuisine": "Mexican", "time": 15, "difficulty": "Easy", "dietary": "Non-Vegetarian", "calories": 380, "rating": 4.5},
            {"name": "Carnitas", "ingredients": ["pork", "orange", "spices", "onions"], "cuisine": "Mexican", "time": 20, "difficulty": "Medium", "dietary": "Non-Vegetarian", "calories": 420, "rating": 4.7},
            {"name": "Chicken Enchiladas", "ingredients": ["tortillas", "chicken", "cheese", "sauce", "vegetables"], "cuisine": "Mexican", "time": 30, "difficulty": "Medium", "dietary": "Non-Vegetarian", "calories": 450, "rating": 4.8},
            {"name": "Beef Barbacoa", "ingredients": ["beef", "chili", "spices", "herbs"], "cuisine": "Mexican", "time": 40, "difficulty": "Hard", "dietary": "Non-Vegetarian", "calories": 380, "rating": 4.4},

            # Japanese Cuisine (Vegetarian)
            {"name": "Edamame", "ingredients": ["soybeans", "salt"], "cuisine": "Japanese", "time": 10, "difficulty": "Easy", "dietary": "Vegetarian", "calories": 120, "rating": 4.3},
            {"name": "Vegetable Tempura", "ingredients": ["vegetables", "batter", "oil"], "cuisine": "Japanese", "time": 15, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 280, "rating": 4.5},
            {"name": "Vegetable Sushi", "ingredients": ["rice", "seaweed", "cucumber", "avocado", "carrot"], "cuisine": "Japanese", "time": 20, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 220, "rating": 4.6},
            {"name": "Miso Soup", "ingredients": ["miso paste", "tofu", "seaweed", "green onions"], "cuisine": "Japanese", "time": 30, "difficulty": "Easy", "dietary": "Vegetarian", "calories": 80, "rating": 4.4},
            {"name": "Vegetable Ramen", "ingredients": ["noodles", "vegetable broth", "vegetables", "tofu"], "cuisine": "Japanese", "time": 40, "difficulty": "Medium", "dietary": "Vegetarian", "calories": 380, "rating": 4.7},

            # Japanese Cuisine (Non-Vegetarian)
            {"name": "Chicken Teriyaki", "ingredients": ["chicken", "teriyaki sauce", "rice", "vegetables"], "cuisine": "Japanese", "time": 10, "difficulty": "Easy", "dietary": "Non-Vegetarian", "calories": 320, "rating": 4.7},
            {"name": "Salmon Sushi", "ingredients": ["rice", "seaweed", "salmon", "avocado"], "cuisine": "Japanese", "time": 15, "difficulty": "Medium", "dietary": "Non-Vegetarian", "calories": 280, "rating": 4.8},
            {"name": "Beef Sukiyaki", "ingredients": ["beef", "tofu", "vegetables", "noodles", "sauce"], "cuisine": "Japanese", "time": 20, "difficulty": "Medium", "dietary": "Non-Vegetarian", "calories": 420, "rating": 4.6},
            {"name": "Tonkatsu", "ingredients": ["pork", "breading", "cabbage", "sauce"], "cuisine": "Japanese", "time": 30, "difficulty": "Medium", "dietary": "Non-Vegetarian", "calories": 480, "rating": 4.5},
            {"name": "Chicken Ramen", "ingredients": ["noodles", "chicken broth", "chicken", "vegetables", "egg"], "cuisine": "Japanese", "time": 40, "difficulty": "Hard", "dietary": "Non-Vegetarian", "calories": 450, "rating": 4.9},
        ]

        self.recipes_df = pd.DataFrame(recipes_data)

    def preprocess_data(self):
        """Preprocess data using pandas and numpy"""
        # Encode categorical variables
        le_difficulty = LabelEncoder()
        le_cuisine = LabelEncoder()
        le_dietary = LabelEncoder()

        self.recipes_df['difficulty_encoded'] = le_difficulty.fit_transform(self.recipes_df['difficulty'])
        self.recipes_df['cuisine_encoded'] = le_cuisine.fit_transform(self.recipes_df['cuisine'])
        self.recipes_df['dietary_encoded'] = le_dietary.fit_transform(self.recipes_df['dietary'])


        self.recipes_df['complexity_score'] = (
            self.recipes_df['time'] * 0.4 +
            self.recipes_df['difficulty_encoded'] * 30 +
            self.recipes_df['rating'] * 10 +
            self.recipes_df['calories'] * 0.01
        )


        self.recipes_df['ingredient_count'] = self.recipes_df['ingredients'].apply(len)
        self.recipes_df['nutritional_density'] = self.recipes_df['calories'] / self.recipes_df['ingredient_count']

    def recommend_recipes(self, cuisine=None, max_time=None, dietary=None, min_rating=0, max_calories=None):
        """Enhanced recommendation system using pandas filtering"""
        filtered_df = self.recipes_df.copy()


        if cuisine:
            filtered_df = filtered_df[filtered_df['cuisine'].str.lower() == cuisine.lower()]
        if max_time:
            filtered_df = filtered_df[filtered_df['time'] <= max_time]
        if dietary:
            filtered_df = filtered_df[filtered_df['dietary'].str.lower() == dietary.lower()]
        if max_calories:
            filtered_df = filtered_df[filtered_df['calories'] <= max_calories]

        filtered_df = filtered_df[filtered_df['rating'] >= min_rating]


        filtered_df['recommendation_score'] = (
            filtered_df['rating'] * 0.4 +
            (1 / filtered_df['time']) * 25 +
            (1 / filtered_df['complexity_score']) * 20 +
            (1 / filtered_df['calories']) * 15
        )


        filtered_df = filtered_df.sort_values('recommendation_score', ascending=False)

        return filtered_df

    def analyze_cuisine_stats(self):
        """Statistical analysis using pandas and numpy"""
        stats = self.recipes_df.groupby('cuisine').agg({
            'time': ['mean', 'std', 'min', 'max'],
            'rating': ['mean', 'std'],
            'calories': ['mean', 'std'],
            'ingredient_count': ['mean', 'std'],
            'complexity_score': ['mean', 'std']
        }).round(2)

        return stats

    def plot_cuisine_analysis(self):
        """Create comprehensive visualizations using matplotlib"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))


        avg_time = self.recipes_df.groupby('cuisine')['time'].mean().sort_values()
        colors = plt.cm.Set3(np.linspace(0, 1, len(avg_time)))
        axes[0,0].bar(avg_time.index, avg_time.values, color=colors)
        axes[0,0].set_title('Average Cooking Time by Cuisine', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('Time (minutes)')
        axes[0,0].tick_params(axis='x', rotation=45)


        cuisine_ratings = [self.recipes_df[self.recipes_df['cuisine'] == cuisine]['rating']
                          for cuisine in self.recipes_df['cuisine'].unique()]
        box_plot = axes[0,1].boxplot(cuisine_ratings, labels=self.recipes_df['cuisine'].unique())
        axes[0,1].set_title('Rating Distribution by Cuisine', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('Rating')
        axes[0,1].tick_params(axis='x', rotation=45)


        difficulty_counts = self.recipes_df['difficulty'].value_counts()
        axes[1,0].pie(difficulty_counts.values, labels=difficulty_counts.index, autopct='%1.1f%%',
                     colors=['#ff9999', '#66b3ff', '#99ff99'])
        axes[1,0].set_title('Difficulty Level Distribution', fontsize=14, fontweight='bold')


        numeric_cols = ['time', 'rating', 'calories', 'ingredient_count', 'complexity_score']
        corr_matrix = self.recipes_df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1],
                   square=True, cbar_kws={"shrink": .8})
        axes[1,1].set_title('Correlation Heatmap', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.show()

    def plot_regression_analysis(self, cuisine_input, dietary_input):
        """Enhanced regression analysis with mathematical operations"""
        filtered_recipes = self.recipes_df[
            (self.recipes_df['cuisine'].str.lower() == cuisine_input.lower()) &
            (self.recipes_df['dietary'].str.lower() == dietary_input.lower())
        ]

        if len(filtered_recipes) == 0:
            print(f"No recipes found for {cuisine_input} cuisine with {dietary_input} dietary preference.")
            return


        X = filtered_recipes[['time', 'ingredient_count']].values
        y = filtered_recipes['calories'].values

        # Perform linear regression using numpy
        if len(X) > 1:  # Need at least 2 points for regression
            # Calculate regression coefficients manually using numpy
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
            coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

            # Create mesh for 3D plot
            time_range = np.linspace(filtered_recipes['time'].min(), filtered_recipes['time'].max(), 10)
            ingredient_range = np.linspace(filtered_recipes['ingredient_count'].min(),
                                         filtered_recipes['ingredient_count'].max(), 10)
            Time, Ingredients = np.meshgrid(time_range, ingredient_range)
            Calories = coefficients[0] + coefficients[1] * Time + coefficients[2] * Ingredients

            # Create 3D plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Scatter plot of actual data
            scatter = ax.scatter(filtered_recipes['time'],
                               filtered_recipes['ingredient_count'],
                               filtered_recipes['calories'],
                               c=filtered_recipes['rating'], cmap='viridis', s=100, alpha=0.8)

            # Surface plot of regression plane
            surface = ax.plot_surface(Time, Ingredients, Calories, alpha=0.3, cmap='coolwarm')

            ax.set_xlabel('Cooking Time (min)')
            ax.set_ylabel('Number of Ingredients')
            ax.set_zlabel('Calories')
            ax.set_title(f'3D Regression: Time & Ingredients vs Calories\n{cuisine_input} Cuisine ({dietary_input})',
                        fontsize=14, fontweight='bold')

            # Add colorbar for ratings
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
            cbar.set_label('Recipe Rating')

            plt.show()

            # Print regression equation
            print(f"\nRegression Equation: Calories = {coefficients[0]:.2f} + {coefficients[1]:.2f}*Time + {coefficients[2]:.2f}*Ingredients")
        else:
            print("Not enough data points for regression analysis.")

    def predict_cooking_time(self, difficulty, ingredient_count):
        """Predict cooking time using linear regression"""
        # Prepare data for regression
        X = self.recipes_df[['difficulty_encoded', 'ingredient_count']]
        y = self.recipes_df['time']

        # Train linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict for given inputs
        prediction = model.predict([[difficulty, ingredient_count]])
        return math.ceil(prediction[0])

    def get_nutritional_analysis(self):
        """Perform nutritional analysis using numpy and pandas"""
        analysis = {}

        # Basic statistics
        analysis['avg_calories'] = np.mean(self.recipes_df['calories'])
        analysis['std_calories'] = np.std(self.recipes_df['calories'])
        analysis['avg_ingredients'] = np.mean(self.recipes_df['ingredient_count'])

        # Healthiest recipes (low calories, high rating)
        health_score = (1 / self.recipes_df['calories']) * self.recipes_df['rating'] * 100
        self.recipes_df['health_score'] = health_score
        healthiest_recipes = self.recipes_df.nlargest(5, 'health_score')[['name', 'cuisine', 'calories', 'rating', 'health_score']]

        analysis['healthiest_recipes'] = healthiest_recipes

        return analysis

    def get_youtube_search_link(self, recipe_name):
        """Generate YouTube search link"""
        query = urllib.parse.quote(recipe_name + " recipe")
        return f"https://www.youtube.com/results?search_query={query}"

def main():
    # Initialize the recommender system
    recommender = RecipeRecommender()

    print("=== ENHANCED RECIPE RECOMMENDER SYSTEM ===")
    print("=" * 50)

    print("\n1. DATASET OVERVIEW:")
    print(f"Total recipes: {len(recommender.recipes_df)}")
    print(f"Cuisines available: {', '.join(recommender.recipes_df['cuisine'].unique())}")
    print(f"Dietary options: {', '.join(recommender.recipes_df['dietary'].unique())}")

    print("\n2. STATISTICAL ANALYSIS BY CUISINE:")
    stats = recommender.analyze_cuisine_stats()
    print(stats)

    # Get user input with validation
    print("\n3. PERSONALIZED RECOMMENDATIONS:")
    print("Available cuisines: American, Italian, Indian, Thai, Middle Eastern, Korean, Mexican, Japanese")

    while True:
        cuisine_input = input("Enter cuisine type (or press enter for all): ").strip()
        if not cuisine_input or cuisine_input.title() in recommender.recipes_df['cuisine'].unique():
            break
        print("Invalid cuisine. Please choose from available options.")

    while True:
        time_input = input("Enter maximum cooking time in minutes (or press enter for all): ").strip()
        if not time_input:
            max_time = None
            break
        try:
            max_time = int(time_input)
            if max_time > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

    while True:
        dietary_input = input("Enter dietary preference (Vegetarian/Non-Vegetarian or press enter for all): ").strip()
        if not dietary_input or dietary_input in ['Vegetarian', 'Non-Vegetarian']:
            break
        print("Please enter 'Vegetarian' or 'Non-Vegetarian'")

    while True:
        calories_input = input("Enter maximum calories (or press enter for no limit): ").strip()
        if not calories_input:
            max_calories = None
            break
        try:
            max_calories = int(calories_input)
            if max_calories > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

    # Get recommendations
    recommendations = recommender.recommend_recipes(
        cuisine=cuisine_input if cuisine_input else None,
        max_time=max_time,
        dietary=dietary_input if dietary_input else None,
        min_rating=3.5,
        max_calories=max_calories
    )

    print(f"\n4. RECOMMENDATION RESULTS:")
    print(f"Found {len(recommendations)} recipes matching your criteria:")

    if len(recommendations) > 0:
        for idx, (_, recipe) in enumerate(recommendations.head(10).iterrows(), 1):
            print(f"\n{idx}. {recipe['name']} ⭐{recipe['rating']}")
            print(f"   🕒 {recipe['time']}min | 🥗 {recipe['dietary']} | 🔥 {recipe['calories']} cal")
            print(f"   🎯 Difficulty: {recipe['difficulty']} | 🍽️ Ingredients: {recipe['ingredient_count']}")
            print(f"   📊 Recommendation Score: {recipe['recommendation_score']:.2f}")
            print(f"   🥬 Ingredients: {', '.join(recipe['ingredients'])}")

        # YouTube links for top recommendations
        want_video = input(f"\nWould you like YouTube links for top 3 recipes? (yes/no): ").strip().lower()
        if want_video == 'yes':
            print("\n5. YOUTUBE COOKING LINKS:")
            for idx, (_, recipe) in enumerate(recommendations.head(3).iterrows(), 1):
                video_link = recommender.get_youtube_search_link(recipe['name'])
                print(f"{idx}. {recipe['name']}: {video_link}")
    else:
        print("No recipes found matching your criteria. Try relaxing some filters.")

    # Nutritional analysis
    print("\n6. NUTRITIONAL ANALYSIS:")
    nutrition = recommender.get_nutritional_analysis()
    print(f"Average calories per recipe: {nutrition['avg_calories']:.1f}")
    print(f"Average ingredients per recipe: {nutrition['avg_ingredients']:.1f}")
    print("\nTop 5 Healthiest Recipes:")
    print(nutrition['healthiest_recipes'].to_string(index=False))

    # Prediction example
    print("\n7. COOKING TIME PREDICTION:")
    if len(recommendations) > 0:
        sample_recipe = recommendations.iloc[0]
        pred_time = recommender.predict_cooking_time(
            difficulty=sample_recipe['difficulty_encoded'],
            ingredient_count=sample_recipe['ingredient_count']
        )
        print(f"Based on similar recipes, '{sample_recipe['name']}' should take approximately {pred_time} minutes")

    # Generate comprehensive visualizations
    print("\n8. DATA VISUALIZATION:")
    print("Generating comprehensive analysis plots...")
    recommender.plot_cuisine_analysis()

    # 3D Regression analysis if user provided specific filters
    if cuisine_input and dietary_input:
        print("\n9. ADVANCED REGRESSION ANALYSIS:")
        print("Generating 3D regression plot...")
        recommender.plot_regression_analysis(cuisine_input, dietary_input)

    print("\n" + "=" * 50)
    print("Thank you for using the Enhanced Recipe Recommender System!")
    print("=" * 50)

if __name__ == "__main__":
    main()
