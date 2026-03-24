import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class MoodAnalyzer:
    def __init__(self):
        self.mood_data = []
        self.sentiment_scores = []
        self.timestamps = []
    
    def analyze_text(self, text):
        """Analyze text input and extract sentiment score"""
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        self.sentiment_scores.append(sentiment_score)
        self.timestamps.append(datetime.now())
        return sentiment_score
    
    def get_mood_label(self, score):
        """Convert sentiment score to mood label"""
        if score >= 0.5:
            return 'Very Happy'
        elif score >= 0.1:
            return 'Happy'
        elif score > -0.1:
            return 'Neutral'
        elif score > -0.5:
            return 'Sad'
        else:
            return 'Very Sad'
    
    def generate_trend_visualization(self, days=30):
        """Generate mood trend visualization for specified time period"""
        if not self.sentiment_scores:
            return None
            
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'sentiment': self.sentiment_scores
        })
        
        # Filter for specified time period
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df['timestamp'] >= cutoff_date]
        
        # Calculate daily averages
        daily_avg = df.resample('D', on='timestamp')['sentiment'].mean()
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.plot(daily_avg.index, daily_avg.values, 'b-', linewidth=2)
        plt.fill_between(daily_avg.index, daily_avg.values, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        plt.title('Your Mood Journey Over Time')
        plt.xlabel('Date')
        plt.ylabel('Mood Score')
        plt.grid(True, alpha=0.3)
        
        # Add mood zone labels
        plt.axhspan(0.5, 1.0, alpha=0.2, color='green', label='Very Happy')
        plt.axhspan(0.1, 0.5, alpha=0.2, color='lightgreen', label='Happy')
        plt.axhspan(-0.1, 0.1, alpha=0.2, color='gray', label='Neutral')
        plt.axhspan(-0.5, -0.1, alpha=0.2, color='lightcoral', label='Sad')
        plt.axhspan(-1.0, -0.5, alpha=0.2, color='red', label='Very Sad')
        
        plt.legend()
        return plt
    
    def get_mood_statistics(self):
        """Calculate mood statistics"""
        if not self.sentiment_scores:
            return None
            
        return {
            'average_mood': np.mean(self.sentiment_scores),
            'mood_variance': np.var(self.sentiment_scores),
            'most_common_mood': self.get_mood_label(np.median(self.sentiment_scores)),
            'total_entries': len(self.sentiment_scores)
        }
    
    def export_mood_data(self, filepath):
        """Export mood data to CSV"""
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'sentiment_score': self.sentiment_scores,
            'mood_label': [self.get_mood_label(score) for score in self.sentiment_scores]
        })
        df.to_csv(filepath, index=False)
