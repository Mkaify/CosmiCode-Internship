"""
Exploratory Data Analysis (EDA) for Chatbot Dataset

This module provides comprehensive analysis of the chatbot dataset including:
- Dataset overview and statistics
- Intent distribution analysis
- Text pattern analysis
- Vocabulary and linguistic features
- Data quality assessment
- Advanced visualizations and insights

Features:
- Comprehensive statistical analysis
- Advanced visualization techniques
- Text analytics and NLP insights
- Data quality reporting
- Interactive analysis tools
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from textstat import flesch_reading_ease, flesch_kincaid_grade
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

class ChatbotEDA:
    """
    Comprehensive Exploratory Data Analysis for Chatbot Dataset
    """
    
    def __init__(self, json_file):
        """
        Initialize EDA with dataset
        
        Args:
            json_file: Path to the chatbot data JSON file
        """
        self.json_file = json_file
        self.data = None
        self.intents_df = None
        self.patterns_df = None
        self.load_and_prepare_data()
        
    def load_and_prepare_data(self):
        """
        Load JSON data and prepare DataFrames for analysis
        """
        print(f"Loading dataset from {self.json_file}...")
        
        try:
            with open(self.json_file, 'r', encoding='utf-8') as file:
                self.data = json.load(file)
        except FileNotFoundError:
            print(f"Error: File {self.json_file} not found!")
            return
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {self.json_file}")
            return
        
        # Prepare patterns DataFrame
        patterns_data = []
        for intent in self.data['intents']:
            tag = intent['tag']
            for pattern in intent['patterns']:
                patterns_data.append({
                    'tag': tag,
                    'pattern': pattern,
                    'pattern_length': len(pattern.split()),
                    'char_count': len(pattern),
                    'word_count': len(pattern.split()),
                    'response_count': len(intent['responses'])
                })
        
        self.patterns_df = pd.DataFrame(patterns_data)
        
        # Prepare intents DataFrame
        intents_data = []
        for intent in self.data['intents']:
            intents_data.append({
                'tag': intent['tag'],
                'pattern_count': len(intent['patterns']),
                'response_count': len(intent['responses']),
                'avg_pattern_length': np.mean([len(p.split()) for p in intent['patterns']]),
                'avg_response_length': np.mean([len(r.split()) for r in intent['responses']])
            })
        
        self.intents_df = pd.DataFrame(intents_data)
        
        print("Data loaded and prepared successfully!")
        print(f"  Total intents: {len(self.data['intents'])}")
        print(f"  Total patterns: {len(self.patterns_df)}")
    
    def basic_statistics(self):
        """
        Generate basic statistics about the dataset
        """
        print("\n" + "="*60)
        print("BASIC DATASET STATISTICS")
        print("="*60)
        
        # Overall statistics
        total_intents = len(self.data['intents'])
        total_patterns = len(self.patterns_df)
        total_responses = sum([len(intent['responses']) for intent in self.data['intents']])
        
        print(f"Dataset Overview:")
        print(f"  Total Intents: {total_intents}")
        print(f"  Total Patterns: {total_patterns}")
        print(f"  Total Responses: {total_responses}")
        print(f"  Average Patterns per Intent: {total_patterns / total_intents:.2f}")
        print(f"  Average Responses per Intent: {total_responses / total_intents:.2f}")
        
        # Pattern statistics
        print(f"\nPattern Statistics:")
        print(f"  Average Pattern Length: {self.patterns_df['word_count'].mean():.2f} words")
        print(f"  Median Pattern Length: {self.patterns_df['word_count'].median():.2f} words")
        print(f"  Min Pattern Length: {self.patterns_df['word_count'].min()} words")
        print(f"  Max Pattern Length: {self.patterns_df['word_count'].max()} words")
        print(f"  Standard Deviation: {self.patterns_df['word_count'].std():.2f} words")
        
        # Intent distribution
        print(f"\nIntent Distribution:")
        intent_counts = self.patterns_df['tag'].value_counts()
        print(f"  Most Common Intent: {intent_counts.index[0]} ({intent_counts.iloc[0]} patterns)")
        print(f"  Least Common Intent: {intent_counts.index[-1]} ({intent_counts.iloc[-1]} patterns)")
        print(f"  Balance Ratio (Max/Min): {intent_counts.iloc[0] / intent_counts.iloc[-1]:.2f}")
        
        # Character-level statistics
        print(f"\nCharacter-Level Statistics:")
        print(f"  Average Character Count: {self.patterns_df['char_count'].mean():.2f}")
        print(f"  Total Characters: {self.patterns_df['char_count'].sum()}")
        
    def intent_analysis(self):
        """
        Detailed analysis of intents
        """
        print("\n" + "="*60)
        print("INTENT ANALYSIS")
        print("="*60)
        
        # Intent distribution table
        intent_stats = self.patterns_df.groupby('tag').agg({
            'pattern': 'count',
            'word_count': ['mean', 'std', 'min', 'max'],
            'char_count': 'mean'
        }).round(2)
        
        intent_stats.columns = ['Pattern_Count', 'Avg_Words', 'Std_Words', 'Min_Words', 'Max_Words', 'Avg_Chars']
        intent_stats = intent_stats.sort_values('Pattern_Count', ascending=False)
        
        print("Intent Statistics (sorted by pattern count):")
        print(intent_stats.to_string())
        
        # Class balance analysis
        pattern_counts = self.patterns_df['tag'].value_counts()
        balance_coefficient = pattern_counts.std() / pattern_counts.mean()
        
        print(f"\nClass Balance Analysis:")
        print(f"  Balance Coefficient: {balance_coefficient:.3f}")
        if balance_coefficient < 0.3:
            print("  Dataset is well-balanced ✓")
        elif balance_coefficient < 0.6:
            print("  Dataset is moderately balanced ⚠")
        else:
            print("  Dataset is imbalanced ⚠")
        
        # Response analysis
        print(f"\nResponse Analysis:")
        for intent in self.data['intents']:
            responses = intent['responses']
            avg_response_length = np.mean([len(r.split()) for r in responses])
            print(f"  {intent['tag']}: {len(responses)} responses, avg {avg_response_length:.1f} words")
    
    def text_analysis(self):
        """
        Comprehensive text analysis
        """
        print("\n" + "="*60)
        print("TEXT ANALYSIS")
        print("="*60)
        
        # Combine all patterns for analysis
        all_patterns = ' '.join(self.patterns_df['pattern'].tolist())
        all_words = word_tokenize(all_patterns.lower())
        
        # Vocabulary analysis
        vocabulary = set(all_words)
        word_freq = Counter(all_words)
        
        print(f"Vocabulary Analysis:")
        print(f"  Total Words: {len(all_words)}")
        print(f"  Unique Words: {len(vocabulary)}")
        print(f"  Vocabulary Richness: {len(vocabulary) / len(all_words):.3f}")
        
        # Most common words
        print(f"\nMost Common Words:")
        for word, count in word_freq.most_common(10):
            print(f"  {word}: {count}")
        
        # Least common words (appear only once)
        hapax_legomena = [word for word, count in word_freq.items() if count == 1]
        print(f"\nRare Words (appearing once): {len(hapax_legomena)}")
        print(f"  Examples: {hapax_legomena[:10]}")
        
        # N-gram analysis
        print(f"\nN-gram Analysis:")
        
        # Bigrams
        bigrams = list(ngrams(all_words, 2))
        bigram_freq = Counter(bigrams)
        print(f"  Most Common Bigrams:")
        for bigram, count in bigram_freq.most_common(5):
            print(f"    {' '.join(bigram)}: {count}")
        
        # Trigrams
        trigrams = list(ngrams(all_words, 3))
        trigram_freq = Counter(trigrams)
        print(f"  Most Common Trigrams:")
        for trigram, count in trigram_freq.most_common(5):
            print(f"    {' '.join(trigram)}: {count}")
        
        # Readability analysis
        print(f"\nReadability Analysis:")
        try:
            reading_ease = flesch_reading_ease(all_patterns)
            grade_level = flesch_kincaid_grade(all_patterns)
            print(f"  Flesch Reading Ease: {reading_ease:.2f}")
            print(f"  Flesch-Kincaid Grade Level: {grade_level:.2f}")
        except:
            print("  Readability analysis not available")
    
    def data_quality_assessment(self):
        """
        Assess data quality and identify potential issues
        """
        print("\n" + "="*60)
        print("DATA QUALITY ASSESSMENT")
        print("="*60)
        
        quality_issues = []
        
        # Check for empty patterns
        empty_patterns = self.patterns_df[self.patterns_df['pattern'].str.strip() == '']
        if len(empty_patterns) > 0:
            quality_issues.append(f"Found {len(empty_patterns)} empty patterns")
        
        # Check for very short patterns
        short_patterns = self.patterns_df[self.patterns_df['word_count'] <= 1]
        if len(short_patterns) > 0:
            quality_issues.append(f"Found {len(short_patterns)} very short patterns (≤1 word)")
        
        # Check for very long patterns
        long_patterns = self.patterns_df[self.patterns_df['word_count'] >= 20]
        if len(long_patterns) > 0:
            quality_issues.append(f"Found {len(long_patterns)} very long patterns (≥20 words)")
        
        # Check for duplicate patterns
        duplicates = self.patterns_df[self.patterns_df.duplicated(subset=['pattern'], keep=False)]
        if len(duplicates) > 0:
            quality_issues.append(f"Found {len(duplicates)} duplicate patterns")
        
        # Check for patterns with special characters
        special_char_pattern = self.patterns_df[self.patterns_df['pattern'].str.contains(r'[^a-zA-Z0-9\s\'\-\?\!\.]')]
        if len(special_char_pattern) > 0:
            quality_issues.append(f"Found {len(special_char_pattern)} patterns with special characters")
        
        # Check intent balance
        intent_counts = self.patterns_df['tag'].value_counts()
        if intent_counts.max() / intent_counts.min() > 3:
            quality_issues.append(f"Intent imbalance detected (ratio: {intent_counts.max() / intent_counts.min():.1f})")
        
        # Report quality assessment
        if quality_issues:
            print("Quality Issues Found:")
            for issue in quality_issues:
                print(f"  ⚠ {issue}")
        else:
            print("✓ No major quality issues detected!")
        
        # Recommendations
        print(f"\nRecommendations:")
        if len(short_patterns) > 0:
            print("  - Consider expanding very short patterns for better training")
        if len(long_patterns) > 0:
            print("  - Consider splitting very long patterns into shorter ones")
        if intent_counts.max() / intent_counts.min() > 2:
            print("  - Consider balancing intent distribution by adding more patterns to underrepresented intents")
        if len(duplicates) > 0:
            print("  - Remove or modify duplicate patterns to increase diversity")
    
    def create_visualizations(self):
        """
        Create comprehensive visualizations
        """
        print("\nCreating comprehensive visualizations...")
        
        # Set up the plotting area
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Intent Distribution (Bar Plot)
        plt.subplot(3, 3, 1)
        intent_counts = self.patterns_df['tag'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(intent_counts)))
        bars = plt.bar(range(len(intent_counts)), intent_counts.values, color=colors)
        plt.xlabel('Intent')
        plt.ylabel('Number of Patterns')
        plt.title('Intent Distribution', fontweight='bold')
        plt.xticks(range(len(intent_counts)), intent_counts.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, intent_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(value), ha='center', va='bottom', fontsize=8)
        
        # 2. Pattern Length Distribution
        plt.subplot(3, 3, 2)
        plt.hist(self.patterns_df['word_count'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(self.patterns_df['word_count'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.patterns_df["word_count"].mean():.1f}')
        plt.axvline(self.patterns_df['word_count'].median(), color='green', linestyle='--', 
                   label=f'Median: {self.patterns_df["word_count"].median():.1f}')
        plt.xlabel('Pattern Length (words)')
        plt.ylabel('Frequency')
        plt.title('Pattern Length Distribution', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Intent Distribution (Pie Chart)
        plt.subplot(3, 3, 3)
        plt.pie(intent_counts.values, labels=intent_counts.index, autopct='%1.1f%%', 
               startangle=90, colors=colors)
        plt.title('Intent Distribution (Percentage)', fontweight='bold')
        
        # 4. Box Plot of Pattern Lengths by Intent
        plt.subplot(3, 3, 4)
        box_data = [self.patterns_df[self.patterns_df['tag'] == tag]['word_count'].values 
                   for tag in intent_counts.index]
        plt.boxplot(box_data, labels=intent_counts.index)
        plt.xlabel('Intent')
        plt.ylabel('Pattern Length (words)')
        plt.title('Pattern Length Distribution by Intent', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 5. Character Count vs Word Count Scatter
        plt.subplot(3, 3, 5)
        scatter = plt.scatter(self.patterns_df['word_count'], self.patterns_df['char_count'], 
                            alpha=0.6, c=self.patterns_df['tag'].astype('category').cat.codes, 
                            cmap='tab20')
        plt.xlabel('Word Count')
        plt.ylabel('Character Count')
        plt.title('Word Count vs Character Count', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(self.patterns_df['word_count'], self.patterns_df['char_count'], 1)
        p = np.poly1d(z)
        plt.plot(self.patterns_df['word_count'], p(self.patterns_df['word_count']), 
                "r--", alpha=0.8, label=f'Trend line')
        plt.legend()
        
        # 6. Heatmap of Intent vs Pattern Length
        plt.subplot(3, 3, 6)
        # Create a pivot table for the heatmap
        length_bins = pd.cut(self.patterns_df['word_count'], bins=5, labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
        heatmap_data = pd.crosstab(self.patterns_df['tag'], length_bins)
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Intent vs Pattern Length Distribution', fontweight='bold')
        plt.xlabel('Pattern Length Category')
        plt.ylabel('Intent')
        
        # 7. Response Count Analysis
        plt.subplot(3, 3, 7)
        response_counts = [len(intent['responses']) for intent in self.data['intents']]
        intent_names = [intent['tag'] for intent in self.data['intents']]
        plt.bar(range(len(response_counts)), response_counts, alpha=0.7, color='lightcoral')
        plt.xlabel('Intent')
        plt.ylabel('Number of Responses')
        plt.title('Response Count by Intent', fontweight='bold')
        plt.xticks(range(len(intent_names)), intent_names, rotation=45, ha='right')
        
        # 8. Vocabulary Growth Curve
        plt.subplot(3, 3, 8)
        all_patterns = self.patterns_df['pattern'].tolist()
        vocabulary_growth = []
        unique_words = set()
        
        for i, pattern in enumerate(all_patterns):
            words = set(word_tokenize(pattern.lower()))
            unique_words.update(words)
            vocabulary_growth.append(len(unique_words))
        
        plt.plot(range(1, len(vocabulary_growth) + 1), vocabulary_growth, 'b-', linewidth=2)
        plt.xlabel('Number of Patterns Processed')
        plt.ylabel('Vocabulary Size')
        plt.title('Vocabulary Growth Curve', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 9. Pattern Complexity Analysis
        plt.subplot(3, 3, 9)
        # Calculate complexity as ratio of unique words to total words
        complexity_scores = []
        for pattern in self.patterns_df['pattern']:
            words = word_tokenize(pattern.lower())
            if len(words) > 0:
                complexity = len(set(words)) / len(words)
            else:
                complexity = 0
            complexity_scores.append(complexity)
        
        self.patterns_df['complexity'] = complexity_scores
        
        plt.hist(complexity_scores, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(np.mean(complexity_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(complexity_scores):.2f}')
        plt.xlabel('Pattern Complexity (Unique Words / Total Words)')
        plt.ylabel('Frequency')
        plt.title('Pattern Complexity Distribution', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_word_cloud(self):
        """
        Create word clouds for different intents
        """
        print("Creating word clouds...")
        
        # Overall word cloud
        all_text = ' '.join(self.patterns_df['pattern'])
        
        plt.figure(figsize=(15, 10))
        
        # Overall word cloud
        plt.subplot(2, 3, 1)
        wordcloud = WordCloud(width=400, height=300, background_color='white').generate(all_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Overall Word Cloud', fontweight='bold')
        
        # Word clouds for top 5 intents
        top_intents = self.patterns_df['tag'].value_counts().head(5)
        
        for i, (intent, count) in enumerate(top_intents.items(), 2):
            plt.subplot(2, 3, i)
            intent_text = ' '.join(self.patterns_df[self.patterns_df['tag'] == intent]['pattern'])
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate(intent_text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'{intent.title()} Intent', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_plots(self):
        """
        Create interactive plots using Plotly
        """
        print("Creating interactive visualizations...")
        
        # Interactive Intent Distribution
        intent_counts = self.patterns_df['tag'].value_counts()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Intent Distribution', 'Pattern Length by Intent', 
                          'Word vs Character Count', 'Pattern Complexity by Intent'),
            specs=[[{"type": "bar"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "violin"}]]
        )
        
        # Bar plot
        fig.add_trace(
            go.Bar(x=intent_counts.index, y=intent_counts.values, name='Intent Counts'),
            row=1, col=1
        )
        
        # Box plot
        for intent in intent_counts.index:
            intent_data = self.patterns_df[self.patterns_df['tag'] == intent]['word_count']
            fig.add_trace(
                go.Box(y=intent_data, name=intent, showlegend=False),
                row=1, col=2
            )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=self.patterns_df['word_count'],
                y=self.patterns_df['char_count'],
                mode='markers',
                text=self.patterns_df['tag'],
                name='Patterns',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Violin plot
        for intent in intent_counts.index[:5]:  # Top 5 intents
            intent_data = self.patterns_df[self.patterns_df['tag'] == intent]['complexity']
            fig.add_trace(
                go.Violin(y=intent_data, name=intent, showlegend=False),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Interactive Chatbot Dataset Analysis")
        fig.show()
    
    def generate_detailed_report(self):
        """
        Generate a comprehensive EDA report
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE EDA REPORT")
        print("="*80)
        
        # Dataset summary
        self.basic_statistics()
        
        # Intent analysis
        self.intent_analysis()
        
        # Text analysis
        self.text_analysis()
        
        # Data quality assessment
        self.data_quality_assessment()
        
        # Advanced insights
        print("\n" + "="*60)
        print("ADVANCED INSIGHTS")
        print("="*60)
        
        # Pattern diversity analysis
        all_patterns = self.patterns_df['pattern'].str.lower()
        unique_patterns = all_patterns.nunique()
        total_patterns = len(all_patterns)
        diversity_ratio = unique_patterns / total_patterns
        
        print(f"Pattern Diversity:")
        print(f"  Unique Patterns: {unique_patterns}/{total_patterns}")
        print(f"  Diversity Ratio: {diversity_ratio:.3f}")
        
        # Semantic similarity hints
        print(f"\nSemantic Analysis Hints:")
        short_patterns = self.patterns_df[self.patterns_df['word_count'] <= 2]
        question_patterns = self.patterns_df[self.patterns_df['pattern'].str.contains(r'\?')]
        greeting_patterns = self.patterns_df[self.patterns_df['pattern'].str.contains(r'\b(hi|hello|hey)\b', case=False)]
        
        print(f"  Short patterns (≤2 words): {len(short_patterns)} ({len(short_patterns)/total_patterns*100:.1f}%)")
        print(f"  Question patterns: {len(question_patterns)} ({len(question_patterns)/total_patterns*100:.1f}%)")
        print(f"  Greeting patterns: {len(greeting_patterns)} ({len(greeting_patterns)/total_patterns*100:.1f}%)")
        
        # Recommendations for model training
        print(f"\nModel Training Recommendations:")
        print(f"  - Dataset size is {'adequate' if total_patterns >= 200 else 'small'} for training")
        print(f"  - Intent balance is {'good' if self.patterns_df['tag'].value_counts().std() / self.patterns_df['tag'].value_counts().mean() < 0.5 else 'needs attention'}")
        print(f"  - Pattern diversity is {'high' if diversity_ratio > 0.8 else 'moderate' if diversity_ratio > 0.6 else 'low'}")
        
        return {
            'total_intents': len(self.data['intents']),
            'total_patterns': total_patterns,
            'unique_patterns': unique_patterns,
            'diversity_ratio': diversity_ratio,
            'avg_pattern_length': self.patterns_df['word_count'].mean(),
            'intent_balance': self.patterns_df['tag'].value_counts().std() / self.patterns_df['tag'].value_counts().mean()
        }

def demonstrate_eda():
    """
    Demonstrate comprehensive EDA
    """
    print("CHATBOT DATASET - EXPLORATORY DATA ANALYSIS")
    print("=" * 55)
    
    # Initialize EDA
    eda = ChatbotEDA('chatbot_data.json')
    
    if eda.data is None:
        print("Failed to load data. Please ensure chatbot_data.json exists.")
        return None
    
    # Generate comprehensive report
    report = eda.generate_detailed_report()
    
    # Create visualizations
    eda.create_visualizations()
    
    # Create word clouds
    eda.create_word_cloud()
    
    # Create interactive plots (optional)
    try:
        eda.create_interactive_plots()
    except Exception as e:
        print(f"Interactive plots not available: {e}")
    
    print("\n" + "="*80)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("Key findings:")
    print(f"  • Dataset contains {report['total_intents']} intents with {report['total_patterns']} patterns")
    print(f"  • Pattern diversity ratio: {report['diversity_ratio']:.3f}")
    print(f"  • Average pattern length: {report['avg_pattern_length']:.1f} words")
    print(f"  • Intent balance score: {report['intent_balance']:.3f}")
    
    return eda, report

if __name__ == "__main__":
    # Run comprehensive EDA
    eda_analyzer, eda_report = demonstrate_eda()
    
    if eda_analyzer:
        print("\nEDA analysis completed! Ready for model development.") 