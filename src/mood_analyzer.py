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
    
    def add_mood_entry(self, text, timestamp=None):
        """Add a new mood entry with automatic sentiment analysis"""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Analyze sentiment using TextBlob
        analysis = TextBlob(text)
        sentiment_score = analysis.sentiment.polarity
        
        self.mood_data.append(text)
        self.sentiment_scores.append(sentiment_score)
        self.timestamps.append(timestamp)
    
    def get_trend_analysis(self, days=30):
        """Analyze mood trends over specified number of days"""
        if not self.mood_data:
            return None
            
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'sentiment': self.sentiment_scores,
            'text': self.mood_data
        })
        
        # Resample to daily averages
        daily_avg = df.set_index('timestamp')\\
                      .resample('D')['sentiment']\\
                      .mean()\\
                      .fillna(method='ffill')
        
        return daily_avg
    
    def visualize_mood_trend(self, days=30):
        """Generate visualization of mood trends"""
        trend_data = self.get_trend_analysis(days)
        
        if trend_data is None:
            return None
        
        plt.figure(figsize=(12, 6))
        plt.plot(trend_data.index, trend_data.values, 'b-', linewidth=2)
        plt.fill_between(trend_data.index, trend_data.values, alpha=0.3)
        
        plt.title('Mood Trend Analysis')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add trend line
        z = np.polyfit(range(len(trend_data)), trend_data.values, 1)
        p = np.poly1d(z)
        plt.plot(trend_data.index, p(range(len(trend_data))), 'r--', alpha=0.8)
        
        plt.tight_layout()
        return plt
    
    def get_mood_summary(self):
        """Generate statistical summary of mood data"""
        if not self.sentiment_scores:
            return None
            
        return {
            'average_sentiment': np.mean(self.sentiment_scores),
            'sentiment_variance': np.var(self.sentiment_scores),
            'total_entries': len(self.mood_data),
            'most_positive': max(self.sentiment_scores),
            'most_negative': min(self.sentiment_scores)
        }
    
    def export_mood_data(self, format='csv'):
        """Export mood data to specified format"""
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'sentiment': self.sentiment_scores,
            'text': self.mood_data
        })
        
        if format == 'csv':
            return df.to_csv(index=False)
        elif format == 'json':
            return df.to_json(orient='records')
        else:
            raise ValueError('Unsupported export format')
