import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import warnings
import os
warnings.filterwarnings('ignore')

# Создаем папку для графиков
os.makedirs('wine_plots', exist_ok=True)

# 1. Загрузка данных
print("=== 1. ЗАГРУЗКА ДАННЫХ ===")
with open('winequality-red.csv', 'r') as file:
    content = file.read()

df = pd.read_csv(StringIO(content), sep=';')
print(f"Размер данных: {df.shape}")
print(f"Столбцы: {list(df.columns)}")
print("\nПервые 5 строк:")
print(df.head())
print()

# 2. Проверка на пропуски
print("=== 2. ПРОВЕРКА НА ПРОПУСКИ ===")
missing_values = df.isnull().sum()
total_cells = df.shape[0] * df.shape[1]
total_missing = missing_values.sum()

print("Пропуски по столбцам:")
for col, missing_count in missing_values.items():
    print(f"  {col}: {missing_count} пропусков")

print(f"\nВсего ячеек: {total_cells}")
print(f"Всего пропусков: {total_missing}")
print(f"Процент пропусков: {round((total_missing/total_cells)*100, 2)}%")

rows_with_missing = df.isnull().any(axis=1).sum()
print(f"Строк с пропусками: {rows_with_missing}")
print()

# 3. Преобразование в бинарные классы
print("=== 3. ПРЕОБРАЗОВАНИЕ В БИНАРНЫЕ КЛАССЫ ===")
df['binary_quality'] = (df['quality'] >= 6).astype(int)
class_counts = df['binary_quality'].value_counts()

print("Распределение исходного качества:")
print(df['quality'].value_counts().sort_index())
print("\nРаспределение бинарных классов:")
print(f"Плохие вина (0): {class_counts[0]} ({class_counts[0]/len(df)*100:.1f}%)")
print(f"Хорошие вина (1): {class_counts[1]} ({class_counts[1]/len(df)*100:.1f}%)")
print()

# 4. Анализ выбросов по качеству
print("=== 4. АНАЛИЗ ВЫБРОСОВ ПО КАЧЕСТВУ ===")
Q1 = df['quality'].quantile(0.25)
Q3 = df['quality'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['quality'] < lower_bound) | (df['quality'] > upper_bound)]

print(f"Q1 (25-й процентиль): {Q1}")
print(f"Q3 (75-й процентиль): {Q3}")
print(f"IQR: {IQR}")
print(f"Нижняя граница выбросов: {lower_bound:.2f}")
print(f"Верхняя граница выбросов: {upper_bound:.2f}")
print(f"Количество выбросов: {len(outliers)}")

# Удаление выбросов
df_clean = df[(df['quality'] >= lower_bound) & (df['quality'] <= upper_bound)].copy()
print(f"Данные до очистки: {df.shape}")
print(f"Данные после очистки: {df_clean.shape}")
print(f"Удалено записей: {len(df) - len(df_clean)}")
print()

# Медианы по признакам
print("=== МЕДИАНЫ ПО ПРИЗНАКАМ ===")
medians = df_clean.median(numeric_only=True)
for col, median_val in medians.items():
    print(f"{col:25}: {median_val:.4f}")
print()

# 5. ВИЗУАЛИЗАЦИЯ И СОХРАНЕНИЕ ГРАФИКОВ
print("=== 5. СОХРАНЕНИЕ ГРАФИКОВ ===")

# Настройка стиля графиков
plt.style.use('default')
sns.set_palette("husl")

# 5.1 График распределения качества
plt.figure(figsize=(10, 6))
sns.histplot(df_clean['quality'], kde=True, bins=10, color='skyblue')
plt.title('Распределение качества вина', fontweight='bold')
plt.xlabel('Качество')
plt.ylabel('Частота')
plt.grid(True, alpha=0.3)
plt.savefig('wine_plots/01_quality_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Сохранен: wine_plots/01_quality_distribution.png")

# 5.2 Баланс бинарных классов
plt.figure(figsize=(8, 6))
class_counts_clean = df_clean['binary_quality'].value_counts()
plt.pie(class_counts_clean.values, labels=['Плохое (0)', 'Хорошее (1)'], 
        autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'], startangle=90)
plt.title('Баланс бинарных классов', fontweight='bold')
plt.savefig('wine_plots/02_class_balance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Сохранен: wine_plots/02_class_balance.png")

# 5.3 Ящик с усами по качеству
plt.figure(figsize=(8, 6))
sns.boxplot(y=df_clean['quality'], color='lightyellow')
plt.title('Ящик с усами - Качество вина', fontweight='bold')
plt.ylabel('Качество')
plt.savefig('wine_plots/03_quality_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Сохранен: wine_plots/03_quality_boxplot.png")

# 5.4 Ящики с усами для всех признаков
plt.figure(figsize=(14, 8))
features_for_plot = [col for col in df_clean.columns if col not in ['quality', 'binary_quality']]
df_clean[features_for_plot].boxplot()
plt.title('Ящики с усами для всех признаков', fontweight='bold')
plt.xticks(rotation=45)
plt.savefig('wine_plots/04_all_features_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Сохранен: wine_plots/04_all_features_boxplot.png")

# 5.5 Матрица корреляции
plt.figure(figsize=(12, 10))
correlation_matrix = df_clean.corr(numeric_only=True)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
            center=0, square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Матрица корреляции признаков вина', fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.savefig('wine_plots/05_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Сохранен: wine_plots/05_correlation_matrix.png")

# 5.6 Корреляции с качеством
plt.figure(figsize=(10, 6))
quality_correlations = correlation_matrix['quality'].drop('quality').sort_values(key=abs, ascending=False)
bars = plt.barh(quality_correlations.index, quality_correlations.values,
                color=['red' if x < 0 else 'green' for x in quality_correlations.values])
plt.xlabel('Коэффициент корреляции')
plt.title('Корреляция признаков с качеством вина', fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

for bar, value in zip(bars, quality_correlations.values):
    plt.text(bar.get_width() + (0.01 if value >= 0 else -0.03), 
             bar.get_y() + bar.get_height()/2, f'{value:.3f}', 
             ha='left' if value >= 0 else 'right', va='center', fontweight='bold')

plt.savefig('wine_plots/06_quality_correlations.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Сохранен: wine_plots/06_quality_correlations.png")

# 6. Детальная визуализация распределений признаков
print("=== 6. РАСПРЕДЕЛЕНИЯ ПРИЗНАКОВ ===")

features = [col for col in df_clean.columns if col != 'binary_quality']
num_features = len(features)

# Создаем несколько графиков для распределений
for i in range(0, num_features, 6):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for j in range(6):
        if i + j < num_features:
            feature = features[i + j]
            ax = axes[j]
            
            # Гистограмма с KDE
            sns.histplot(df_clean[feature], kde=True, ax=ax, alpha=0.7)
            
            # Статистики
            mean_val = df_clean[feature].mean()
            median_val = df_clean[feature].median()
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, label=f'Среднее: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=1, label=f'Медиана: {median_val:.2f}')
            
            ax.set_title(f'Распределение {feature}', fontweight='bold')
            ax.set_xlabel(feature)
            ax.set_ylabel('Частота')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    # Скрываем лишние субплоги
    for j in range(num_features - i, 6):
        if j < len(axes):
            fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(f'wine_plots/07_feature_distributions_{i//6 + 1}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранен: wine_plots/07_feature_distributions_{i//6 + 1}.png")

# 7. Парные зависимости для топ-коррелированных признаков
print("=== 7. ПАРНЫЕ ЗАВИСИМОСТИ ===")

top_features = quality_correlations.head(4).index.tolist() + ['quality']
scatter_matrix = sns.pairplot(df_clean[top_features], diag_kind='hist', 
                             plot_kws={'alpha': 0.6, 's': 20},
                             diag_kws={'alpha': 0.7})
scatter_matrix.fig.suptitle('Парные зависимости признаков с наивысшей корреляцией с качеством', 
                           y=1.02, fontweight='bold')
scatter_matrix.savefig('wine_plots/08_pairplot_top_features.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Сохранен: wine_plots/08_pairplot_top_features.png")

# 8. Сводный отчетный график
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.histplot(df_clean['quality'], kde=True, bins=10, color='skyblue')
plt.title('Распределение качества', fontweight='bold')
plt.xlabel('Качество')

plt.subplot(2, 2, 2)
plt.pie(class_counts_clean.values, labels=['Плохое (0)', 'Хорошее (1)'], 
        autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
plt.title('Баланс классов', fontweight='bold')

plt.subplot(2, 2, 3)
sns.boxplot(y=df_clean['quality'], color='lightyellow')
plt.title('Ящик с усами', fontweight='bold')
plt.ylabel('Качество')

plt.subplot(2, 2, 4)
bars = plt.barh(quality_correlations.head(5).index, quality_correlations.head(5).values,
                color=['red' if x < 0 else 'green' for x in quality_correlations.head(5).values])
plt.xlabel('Корреляция')
plt.title('Топ-5 корреляций с качеством', fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('wine_plots/09_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Сохранен: wine_plots/09_summary_plot.png")

# 9. Статистическая сводка
print("=== 8. СТАТИСТИЧЕСКАЯ СВОДКА ===")
print(f"Общее количество наблюдений: {len(df_clean)}")
print(f"Диапазон качества: {df_clean['quality'].min()} - {df_clean['quality'].max()}")
print(f"Среднее качество: {df_clean['quality'].mean():.2f} ± {df_clean['quality'].std():.2f}")

print("\nНаиболее положительно коррелируют с качеством:")
positive_corr = quality_correlations[quality_correlations > 0].head(3)
for feature, corr in positive_corr.items():
    print(f"  {feature}: {corr:.3f}")

print("\nНаиболее отрицательно коррелируют с качеством:")
negative_corr = quality_correlations[quality_correlations < 0].head(3)
for feature, corr in negative_corr.items():
    print(f"  {feature}: {corr:.3f}")

# Сохранение очищенных данных
df_clean.to_csv('winequality-red-cleaned.csv', index=False)
print("\n✓ Очищенные данные сохранены в 'winequality-red-cleaned.csv'")
print("✓ Все графики сохранены в папке 'wine_plots/'")
print(f"✓ Всего сохранено графиков: {len([name for name in os.listdir('wine_plots') if name.endswith('.png')])}")

# Вывод списка сохраненных файлов
print("\nСОХРАНЕННЫЕ ФАЙЛЫ:")
for file in sorted(os.listdir('wine_plots')):
    if file.endswith('.png'):
        file_path = os.path.join('wine_plots', file)
        file_size = os.path.getsize(file_path) / 1024  # Размер в KB
        print(f"  {file} ({file_size:.1f} KB)")
