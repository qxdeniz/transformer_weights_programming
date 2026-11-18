"""Generate conference-ready tables, a Russian draft and standalone scientific figures."""
import csv
import json
from common import ROOT,read_json

def main():
    assert (ROOT/'results/final/access_log.json').exists(),'Final stage must complete first'
    frozen=read_json(ROOT/'freeze.json')
    comparisons=read_json(ROOT/'results/final/comparisons.json')
    selected=[r for r in comparisons if r['model']=='selected']
    paper=ROOT/'paper';paper.mkdir(exist_ok=True)
    attempts=read_json(ROOT/'results/attempts.json')
    with (paper/'validation_attempts.csv').open('w',newline='') as f:
        fields=['candidate','family','validation_correct','wrong_to_right','right_to_wrong','changed_scalars','delta_l2']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in attempts:
            spec=read_json(ROOT/'results'/r['id']/'edit_spec.json')
            w.writerow({'candidate':r['id'],'family':spec['family'],**{k:r[k] for k in fields[2:]}})
    tex=[r'\begin{tabular}{lrrrrrr}',r'\hline',r'Set & $N$ & Before (\%) & After (\%) & $\Delta$ pp & W$\to$R & R$\to$W \\',r'\hline']
    for r in selected:
        name=r['set'].replace('_',r'\_')
        tex.append(f"{name} & {r['n']} & {100*r['before_em']:.3f} & {100*r['after_em']:.3f} & {r['delta_pp']:+.3f} & {r['wrong_to_right']} & {r['right_to_wrong']} "+r'\\')
    tex.extend([r'\hline',r'\end{tabular}'])
    (paper/'results_table.tex').write_text('\n'.join(tex)+'\n')
    full=[]
    for r in selected:
        bm=read_json(ROOT/'results/final/baseline'/f"{r['set']}.metrics.json")
        em=read_json(ROOT/'results/final/selected'/f"{r['set']}.metrics.json")
        full.append({**r,'before_digit_only':bm['upstream_digits_em'],'after_digit_only':em['upstream_digits_em'],
                     'before_numeric':bm['numeric_accuracy'],'after_numeric':em['numeric_accuracy']})
    with (paper/'main_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in full for k in r)))
        w.writeheader();w.writerows(full)
    uniform=next(r for r in full if r['set']=='test_uniform')
    ci=uniform['paired_bootstrap_95ci_pp']
    conclusion=('На основном тесте подтверждено положительное изменение по заранее заданному критерию.' if ci[0]>0
                else 'На основном тесте подтверждено ухудшение.' if ci[1]<0
                else 'Доверительный интервал основного теста включает ноль; улучшение не подтверждено.')
    table=['| Набор | N | До, % | После, % | Δ, п.п. | Исправлено | Испорчено | 95% ДИ Δ |',
           '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in selected:
        interval=r.get('paired_bootstrap_95ci_pp')
        cis=f'[{interval[0]:+.3f}; {interval[1]:+.3f}]' if interval else 'Полный конечный диапазон'
        table.append(f"| {r['set']} | {r['n']} | {100*r['before_em']:.3f} | {100*r['after_em']:.3f} | {r['delta_pp']:+.3f} | {r['wrong_to_right']} | {r['right_to_wrong']} | {cis} |")
    spec=read_json(ROOT/'edit_spec.json')
    draft=f'''# Прямое редактирование весов малого трансформера для исправления ошибок десятизначного сложения

Рабочий текст для студенческой конференции. Авторы, организация и оформление библиографии заполняются перед подачей. Представленные числа — собственные измерения данного локального запуска; принятие статьи и научная новизна не утверждаются.

## Аннотация

Исследована возможность исправления ошибок готового decoder-only трансформера на 1 644 параметра прямой правкой весов без градиентного обучения. Предложен воспроизводимый протокол CarryBench-10: диагностический набор, validation, случайный финальный тест, стресс-тест переносов и исчерпывающий диапазон малых чисел, всего 20 072 примера. Из 12 заранее ограниченных кандидатов по validation выбрана правка {frozen['selected']['changed_scalars']} независимых скаляров ({100*frozen['selected']['fraction']:.3f}% параметров). На основном случайном финальном тесте strict exact match изменился с {100*uniform['before_em']:.3f}% до {100*uniform['after_em']:.3f}%; парная разность составила {uniform['delta_pp']:+.3f} п.п., 95% bootstrap-интервал [{ci[0]:+.3f}; {ci[1]:+.3f}] п.п. {conclusion} Вывод ограничен конкретной моделью и операцией сложения.

Ключевые слова: трансформер, редактирование весов, арифметика, перенос разряда, воспроизводимость.

## 1. Постановка задачи

Проверяется, способен ли LLM-агент улучшить готовую малую арифметическую модель объяснимой записью новых значений в существующие веса. Исходная модель заранее обучена её автором; наш эксперимент не вызывает backward, вычисление градиентов, шаги оптимизатора, SFT, LoRA или дистилляцию. Редактор использует диагностические примеры и validation, поэтому метод описывается как прямое редактирование весов с обратной связью, а не обучение без данных. Работа мотивирована подходом Weight-Space Programming [2], но не воспроизводит эксперимент с Qwen и не заменяет исходную модель аналитическим сумматором.

## 2. Модель и представление данных

Использован checkpoint best.pt репозитория [1], commit 9de34061c8c268e0fc2198e473911a62b152254c. Архитектура: один decoder-блок, размерность 8, две головы внимания, FFN размерности 12, GELU и обучаемые позиционные embeddings. Входной embedding и выходная матрица связаны. Подсчитано 1 644 независимых обучаемых скаляра; фиксированная причинная маска учитывается отдельно. SHA-256 оригинального checkpoint: a919e48dca049119369bec56b76a07a9cb4c804bc94c13aba9ade0c33092c7ec. В зафиксированном дереве не обнаружена явная лицензия кода или весов; право их публичного распространения не утверждается.

Каждое слагаемое дополняется нулями до 10 цифр. Поразрядные пары кодируются токенами Pab от младшего разряда к старшему. Ответ состоит из 11 цифр суммы в обратном порядке и EOS. Вся последовательность генерируется авторегрессионно жадным неизменным методом автора. Архитектура, forward, формат, декодер, число токенов и параметры инференса одинаковы до и после правки. Эталонная сумма вычисляется только генератором данных и оценщиком.

Опубликованный автором результат 99,04% относится к его набору и digit-only метрике. Он не переносится на предлагаемый бенчмарк. Наша основная метрика строже и требует правильного EOS.

## 3. Протокол CarryBench-10

Diagnostic содержит 1 024 примера, validation — 2 048, test_uniform — 5 000, test_stress — 2 000, полный набор всех упорядоченных пар 0…99 — 10 000. Случайные разбиения не пересекаются с учётом перестановки слагаемых. Область двух малых слагаемых зарезервирована для исчерпывающей проверки. Ручные известные примеры в ней отмечены; полностью невиданной она не называется. Независимые seeds, распределения, хеши и правила выбора зафиксированы до правок в protocol.json.

Diagnostic и validation включают равномерные пары, одиночный перенос, цепочки длиной 2–3, 4–6 и 7–10, нулевое слагаемое, ведущие нули, степень десяти и выход в 11-й разряд. Генератор подтверждает реальные переносы точной разрядной арифметикой. Финальный равномерный тест — основной; стрессовый и исчерпывающий — отдельные вторичные проверки. Полное описание распределений приведено в BENCHMARK.md и исходном benchmark_data.py.

Основной показатель — exact match полной 12-токенной цели. Дополнительно сохраняются точность 11 цифр, числовая точность декодера автора и валидность формата. Для одного и того же входа считаются переходы wrong→right и right→wrong. Неопределённость парной разности на случайных наборах оценивается percentile-bootstrap с 20 000 реплик, фиксированным seed и уровнем 95%. Для конечного диапазона 0…99 приводятся точные количества ошибок. Заранее заданное условие подтверждения улучшения — нижняя граница интервала на test_uniform выше нуля; категории и случайные контроли рассматриваются описательно.

## 4. Диагностика и правка

Исходная strict-точность на diagnostic составила 98,535%, на validation — 98,828%. Read-only анализ активаций выявил примеры смешения нужного столбца цифр с соседним в голове внимания 1. В двух диагностических примерах 11 цифр суммы были правильными, но EOS уступал другому токену. В нескольких примерах с ведущими нулями наблюдалась маломаржинальная путаница цифр 0 и 1. Эти наблюдения мотивируют гипотезы и не доказывают универсальную причину ошибок.

Проверены три заранее записанных ограниченных семейства: масштабирование query-матрицы и bias головы 1 коэффициентами 1,05, 1,15 и 1,30; масштабирование строки EOS коэффициентами 1,10, 1,15 и 1,20; аналитическое смещение ln_f.bias в направлении E₀−E₁ с добавлением к разности логитов 0,15, 0,30 или 0,60. Проверены также три заранее определённых комбинации лучших одиночных вариантов. Каждая попытка начиналась с оригинальных весов. Всего выполнено 12 полных проверок validation; общий оптимизатор и случайный поиск кандидатов не использовались.

Выбран кандидат {frozen['selected']['id']}: композиция H1+H2+H3. Точная спецификация — edit_spec.json. Изменено {frozen['selected']['changed_scalars']} независимых значений, абсолютная L2-норма изменения {frozen['selected']['delta_l2']:.8f}, относительная — {frozen['selected']['relative_l2']:.8f}. Список тензоров, коэффициенты и проверки сохранены вместе с весами. На validation правильных ответов стало 2 031 вместо 2 024: 11 исправлений и 4 регрессии. Validation используется только для выбора, поэтому этот результат не является независимым подтверждением.

Проверены сохранение/загрузка без изменения, откат, неизменность shapes, dtype, buffers и связей весов, конечность значений и сохранность элементов вне маски. Три случайных контроля построены с заранее фиксированными seeds на той же фактической маске и с той же L2-нормой с допуском округления float32; ни один seed не отбирался по качеству. Финальные процессы работают без диагностических hooks. Полный журнал неудачных кандидатов сохранён.

## 5. Результаты

{chr(10).join(table)}

{conclusion}

Исходные финальные результаты сохранялись отдельными процессами без просмотра на этапе разработки. До их открытия зафиксированы SHA-256 выбранных весов, edit_spec, протокола и оценщика. После открытия новая правка не выполнялась. Сравнение по цифрам, форматам, категориям и все случайные контроли приведены в results/final и paper/main_results.csv. Ошибки стрессового и малого диапазона оцениваются отдельно и не скрываются общей средней.

## 6. Ограничения

Исследован один checkpoint, одна операция и одна конфигурация генерации; не проверялись другие модели, дробное умножение и общая математическая способность. Обучающий поток автора недоступен: разбиения отложены от редактирования, но отсутствие пересечений с первоначальным обучением не доказано. Упорядоченные пары в исчерпывающем тесте не являются независимыми статистическими наблюдениями; здесь сообщаются конечные числа ошибок. Bootstrap описывает неопределённость выбора тестовых примеров для фиксированной правки и не учитывает вариацию повторных обучений. Три случайных контроля не доказывают превосходство над всем распределением случайных правок. Изменения, исправляющие EOS, могут улучшать strict EM без улучшения арифметических цифр, поэтому метрики приводятся раздельно. Для причинных выводов о механизме нужны более широкие диагностические проверки. Организационное разделение финальных данных не равнозначно технической недоступности одному исследователю. Измерения времени выполнены на одном CPU после одинакового прогрева и не являются полноценным сравнением производительности.

## Источники

1. anadim. Smallest Addition Transformer Codex. https://github.com/anadim/smallest-addition-transformer-codex . Зафиксированный commit и хеши: sources.json.
2. Takehi R., Teoh J., Zhang A., Khattab O. Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights. 2026. https://rikiyat.github.io/blog/weight-space-programming/ . Использован как контекст исследования.
'''
    (paper/'ARTICLE_DRAFT_RU.md').write_text(draft)
    try:plots(paper,comparisons,frozen)
    except ImportError:
        print('Tables and article saved. Install matplotlib to generate figures.')
    print('Paper artifacts:',paper)

def plots(paper,comparisons,frozen):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    def save(fig,name):
        fig.savefig(paper/f'{name}.pdf',bbox_inches='tight')
        fig.savefig(paper/f'{name}.svg',bbox_inches='tight')
        plt.close(fig)
    bm=read_json(ROOT/'results/baseline/diagnostic.metrics.json')
    em=read_json(ROOT/'results'/frozen['selected']['id']/'diagnostic.metrics.json')
    keys=[k for k in bm['by_category'] if k.startswith('source:')]
    x=np.arange(len(keys));fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(x-.18,[100*(1-bm['by_category'][k]['strict_em']) for k in keys],.36,label='Before')
    ax.bar(x+.18,[100*(1-em['by_category'][k]['strict_em']) for k in keys],.36,label='Selected edit')
    ax.set_xticks(x,[k[7:] for k in keys],rotation=35,ha='right');ax.set_ylabel('Strict error rate (%)')
    ax.set_title('Open diagnostic set: descriptive comparison');ax.legend();save(fig,'diagnostic_categories')
    rows=[r for r in comparisons if r['set']=='test_uniform']+[r for r in comparisons if r['set']=='test_stress' and r['model']=='selected']
    fig,ax=plt.subplots(figsize=(7,3.8))
    for i,r in enumerate(rows):
        lo,hi=r['paired_bootstrap_95ci_pp'];d=r['delta_pp']
        ax.errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o',capsize=4,color='#176b87' if r['model']=='selected' else '#777777')
    ax.axvline(0,color='black',linewidth=.8,linestyle='--')
    ax.set_yticks(range(len(rows)),[r['set']+' / '+r['model'] for r in rows]);ax.invert_yaxis()
    ax.set_xlabel('After − before strict EM (percentage points)');ax.set_title('Paired changes and 95% bootstrap intervals')
    save(fig,'paired_final_changes')

if __name__=='__main__':main()
