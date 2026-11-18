"""Standalone scientific figures, tables and a cautious Russian conference draft."""
import csv,hashlib,math
from collections import defaultdict
from utils import ROOT,read_json,read_rows,write_json

def main():
    assert (ROOT/'results/final/access_log.json').exists(),'Test must be opened after freeze first'
    f=read_json(ROOT/'freeze.json');p=read_json(ROOT/'protocol.json');comp=read_json(ROOT/'results/final/comparisons.json');selected=comp[0]
    paper=ROOT/'paper';paper.mkdir(exist_ok=True)
    b=read_rows(ROOT/'results/final/baseline/test.jsonl');a=read_rows(ROOT/'results/final/selected/test.jsonl')
    groups=defaultdict(list)
    length_only=0
    for before,after in zip(b,a):
        length=len(before['good_tokens'])-len(before['bad_tokens'])
        label='good_shorter' if length<0 else 'good_longer' if length>0 else 'same_token_length'
        # Fixed hash tie-break, no random-seed or model-outcome selection.
        length_pref=length<0 if length else hashlib.sha256(before['id'].encode()).digest()[0]%2==0
        length_only+=int(length_pref)
        groups[label].append((before,after,length_pref))
    lengths={k:{'n':len(v),'before_correct':sum(b['correct'] for b,a,l in v),
                   'after_correct':sum(a['correct'] for b,a,l in v),'length_only_correct':sum(l for b,a,l in v),
                   'wrong_to_right':sum(not b['correct'] and a['correct'] for b,a,l in v),
                   'right_to_wrong':sum(b['correct'] and not a['correct'] for b,a,l in v)} for k,v in groups.items()}
    write_json(paper/'length_analysis.json',{'status':'descriptive secondary analysis; not a new selection criterion',
               'length_baseline':'prefer fewer BPE tokens; equal lengths broken by fixed SHA256(id) bit',
               'length_only_correct':length_only,'n':len(b),'groups':lengths})
    attempts=read_json(ROOT/'results/attempts.json')
    with (paper/'validation_attempts.csv').open('w',newline='') as file:
        fields=['candidate','family','validation_correct','wrong_to_right','right_to_wrong','changed_scalars','delta_l2']
        w=csv.DictWriter(file,fieldnames=fields);w.writeheader()
        for record in attempts:
            spec=read_json(ROOT/'results'/record['id']/'edit_spec.json')
            w.writerow({'candidate':record['id'],'family':spec['family'],**{k:record[k] for k in fields[2:]}})
    tex=[r'\begin{tabular}{lrrrrrr}',r'\hline',r'Condition & $N$ & Before (\%) & After (\%) & $\Delta$ pp & W$\to$R & R$\to$W \\',r'\hline']
    for r in comp:tex.append(f"{r['model'].replace('_',chr(92)+'_')} & {r['n']} & {100*r['before_accuracy']:.3f} & {100*r['after_accuracy']:.3f} & {r['delta_pp']:+.3f} & {r['wrong_to_right']} & {r['right_to_wrong']} "+r'\\')
    tex += [r'\hline',r'\end{tabular}'];(paper/'results_table.tex').write_text('\n'.join(tex)+'\n')
    ci=selected['paired_stratified_95ci_pp'];outcome='Подтверждено положительное изменение основной метрики на данном поднаборе.' if ci[0]>0 else ('Наблюдается отрицательное изменение основной метрики.' if ci[1]<0 else 'Положительное изменение не подтверждено: интервал включает ноль.')
    discordant=selected['wrong_to_right']+selected['right_to_wrong']
    exact_p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(selected['wrong_to_right'],selected['right_to_wrong'])+1))/2**discordant) if discordant else 1.
    write_json(paper/'exact_paired_sensitivity.json',{'status':'post-finalization sensitivity analysis; no model selection',
             'test':'two-sided exact binomial test conditional on discordant paired outcomes',
             'discordant_pairs':discordant,'p_value':exact_p,'predeclared_primary':'paired stratified percentile bootstrap'})
    if discordant<25:
        outcome+=f' Дискордантных пар всего {discordant}; дополнительный точный двусторонний парный тест даёт p={exact_p:.3f}. Bootstrap-интервал не следует трактовать как надёжное доказательство устойчивого эффекта при столь малом числе изменений.'
    report=ROOT/'REPORT.md';text=report.read_text()
    if ci[1]<0 and discordant<25:
        text=text.replace('Negative change confirmed on this subset.',
            f'Observed change is negative. The predeclared percentile-bootstrap interval excludes zero, but only {discordant} pairs are discordant. A post-finalization exact paired sensitivity test gives p={exact_p:.3f}; a stable negative effect is not established. This sensitivity analysis does not change model selection or the recorded primary metric.')
        report.write_text(text)
    table=['| Условие | N | До, % | После, % | Δ, п.п. | Исправлено | Испорчено | 95% ДИ Δ |','|---|---:|---:|---:|---:|---:|---:|---|']
    for r in comp:
        lo,hi=r['paired_stratified_95ci_pp'];table.append(f"| {r['model']} | {r['n']} | {100*r['before_accuracy']:.3f} | {100*r['after_accuracy']:.3f} | {r['delta_pp']:+.3f} | {r['wrong_to_right']} | {r['right_to_wrong']} | [{lo:+.3f}; {hi:+.3f}] |")
    lt=['| Длина правильного предложения | N | Верно до | Верно после | Эвристика длины |','|---|---:|---:|---:|---:|']
    for name,v in lengths.items():lt.append(f"| {name} | {v['n']} | {v['before_correct']} | {v['after_correct']} | {v['length_only_correct']} |")
    spec=read_json(ROOT/'edit_spec.json')
    draft=f'''# Прямое редактирование весов сверхмалой языковой модели: контролируемый эксперимент на поднаборе BLiMP

Рабочий текст для студенческой конференции. Авторы и организация заполняются перед подачей. Работа не заявляет новой универсальной методики, принятия статьи или подтверждения результатов авторами исходных ресурсов.

## Аннотация

Исследована возможность изменения качества готовой языковой модели Glimmer-1-Base с 11 856 независимыми параметрами прямой правкой существующих весов без градиентного обучения. Использован заранее выбранный поднабор восьми разделов BLiMP с раздельными диагностическими, валидационными и финальными данными. Проверены {f['attempts_used']} содержательных кандидатов; выбранная правка меняет {f['selected']['changed_scalars']} скаляров ({100*f['selected']['fraction']:.3f}% параметров). На {selected['n']} отложенных минимальных парах точность изменилась с {100*selected['before_accuracy']:.3f}% до {100*selected['after_accuracy']:.3f}%, разность {selected['delta_pp']:+.3f} п.п., 95% парный стратифицированный bootstrap-интервал [{ci[0]:+.3f}; {ci[1]:+.3f}] п.п. {outcome} Отдельно анализируются регрессии, случайные правки и влияние длины токенизации. Результат не трактуется как доказательство улучшения свободной генерации или общей языковой способности.

Ключевые слова: малая языковая модель, редактирование весов, BLiMP, грамматическая приемлемость, воспроизводимость.

## 1. Постановка задачи

В подходе Weight-Space Programming [3] веса рассматриваются как объект прямого анализа и программного изменения. В данной работе проверяется узкий вопрос: можно ли изменить качество уже обученного малого языкового трансформера объяснимой правкой существующих параметров, сохранив архитектуру, токенизатор и метод оценки? Положительный результат не предполагался обязательным. Исходная модель обучена её авторами; наш этап не выполняет backward, вычисление градиентов, оптимизатор, регрессию, SFT, LoRA, RL или дистилляцию. Метод использует диагностическую обратную связь и не называется обучением без данных.

## 2. Модель и данные

Использована Glimmer-1-Base [1], revision 8023a3da67801f03f545674b1a047dae767b667f. Это модель общего назначения, предварительно обученная на FineWeb-Edu, а не арифметический специалист. Архитектура Llama: 2 слоя, скрытая размерность 16, 4 query-головы, одна KV-голова, head_dim=4, SwiGLU ширины 24, RMSNorm и RoPE. Входной embedding и readout связаны. Фактически подсчитано 11 856 независимых параметров; два имени связанной матрицы не удваивают число скаляров. Файл весов занимает 49 440 байт; SHA-256 e61d0b7487386d59d6bf6d62bb9cab28e43b3be18c678f00a9a27cdf41e86c01. В карточке указана MIT. Версии библиотек, источник и конфигурация сохранены.

Для оценки выбран BLiMP [2] — набор минимальных пар английских предложений, различающихся грамматической приемлемостью. До запуска модели выбраны восемь разделов по разнообразию явлений: местоименное согласование, согласование определителя и существительного, два вида субъектно-глагольного согласования, принцип A, отрицательная полярность, wh/that и нерегулярные причастия. Это поднабор BLiMP, а не весь тест из 67 разделов; опубликованные 52,43% полного BLiMP из карточки модели не считаются нашим исходным результатом.

Из исходных данных до измерений удалены точные повторы и любые пары с предложением, уже встречавшимся в другой сохраняемой паре. На раздел выделены 128 diagnostic и 256 validation-примеров, остальные составляют финальный тест. Diagnostic содержит {p['datasets']['diagnostic']['n']} пар, validation — {p['datasets']['validation']['n']}, test — {p['datasets']['test']['n']}. Seeds, размеры, хеши, удалённые дубликаты и критерии зафиксированы в protocol.json до первой экспериментальной правки. Предложения не повторяются между разбиениями, но шаблоны и словарь остаются общими. Точный обучающий срез FineWeb-Edu неизвестен: отсутствие BLiMP-примеров в первоначальном обучении не доказано.

## 3. Метрика и контроль доступа

Языковая модель присваивает каждому предложению суммарную причинную лог-вероятность всех токенов после BOS. EOS не добавляется; нормализация по длине не выполняется. Пара считается правильной, если log P(правильное предложение) строго больше log P(неправильное). Ничьи считаются ошибками. Такое teacher-forced likelihood scoring является стандартным способом проверки минимальных пар и отличается от свободной генерации ответа. Пять жадных продолжений сохраняются как иллюстрации, без включения в критерий выбора.

Исходные test-прогнозы вычислены в отдельном процессе и сохранены без просмотра. Агент использовал только diagnostic и validation. Перед открытием test зафиксированы SHA-256 выбранных весов, edit_spec, протокола и оценщика. После открытия новых правок не выполнялось. Это организационное разделение, а не доказательство недоступности файлов одному исследователю.

Критерий выбора кандидата: максимальное число правильных validation-пар, затем минимум потерянных исходно правильных ответов, изменённых скаляров и L2-нормы изменения. Максимальный бюджет — 12 полных оценок validation. Для финальной разности используется парный bootstrap внутри каждого раздела с исходными весами разделов: 20 000 реплик, seed 94101, 95% интервал. Условие подтверждения положительного результата — нижняя граница интервала общей test-разности выше нуля. Изменения отдельных разделов — описательные вторичные результаты.

## 4. Наблюдения и гипотезы

Исходная точность составила 547/1024 (53,42%) на diagnostic и 1052/2048 (51,37%) на validation. В диагностической свободной генерации обнаружены повторения запятой и двух других токенов. Анализ raw likelihood показал сильное предпочтение короткой токенизации: все 286 diagnostic-пар с более коротким правильным предложением оценены верно, все 309 с более длинным — неверно; среди одинаковых по токенной длине пар верны 261/429. Это ограничивает интерпретацию оценки как грамматической компетентности.

До проверок validation записаны три ограниченных семейства гипотез. H1 уменьшает три строки связанного embedding/readout для повторяющихся токенов коэффициентами 0,95, 0,85 или 0,70. H2 умножает веса финальной RMSNorm на 1,25, 2 или 4, меняя калибровку вероятностей существующих вычислений. H3 умножает выходную матрицу внимания второго слоя на 1,2, 1,5 или 2 для проверки вклада контекста относительно остаточного пути текущего токена. Далее проверяются три заранее определённые комбинации лучших одиночных вариантов, без подгонки новых коэффициентов. Каждая попытка начинается с оригинальных весов. Нормы активаций и повторения мотивируют гипотезы, но сами по себе не доказывают причинный механизм.

Выбран {f['selected']['id']}, семейство {spec['family']}; изменено {f['selected']['changed_scalars']} значений, L2-норма изменения {f['selected']['delta_l2']:.8f}, относительная {f['selected']['relative_l2']:.8f}. Точные формулы и коэффициенты — в edit_spec.json, все успешные и неуспешные попытки — в results/attempts.json. Проверены нулевая правка, откат, сохранность формы и dtype, buffers, связанных весов и элементов вне маски; прогнозы загруженных checkpoint проверены в новых процессах. Диагностические hooks только читают активации и отсутствуют в финальном инференсе.

Три случайных контроля имеют ту же фактическую ненулевую маску и общую L2-норму с допуском округления float32. Seeds зафиксированы заранее, лучший из них не выбирается. Три контроля дают описательное сравнение, а не доказательство превосходства над всем распределением случайных правок.

## 5. Финальные результаты

{chr(10).join(table)}

{outcome} Всего исправлено {selected['wrong_to_right']} исходных ошибок и появилось {selected['right_to_wrong']} регрессий. Детальные таблицы по восьми разделам находятся в REPORT.md и results/final/comparisons.json. Прямое сравнение абсолютной точности с отдельным десятизначным сумматором недопустимо, поскольку задачи и метрики различаются.

### Описательный анализ влияния длины

{chr(10).join(lt)}

Эвристика длины предпочитает предложение с меньшим числом BPE-токенов; ничья разбивается фиксированным битом SHA-256 идентификатора. Она не использует модель или правильные ответы при выборе правила. Этот дополнительный анализ проводится после фиксации весов и не становится новым критерием выбора или подтверждения. Финальная калибровка логитов способна менять log-likelihood предпочтения без изменения argmax-продолжений для фиксированного префикса; такую прибавку нельзя автоматически считать приобретённым знанием грамматики.

## 6. Ограничения и воспроизводимость

Один малый checkpoint и один редактор не определяют эффект на других архитектурах, предобучениях и языках. Выбор модели и поднабора — исследовательский дизайн, а не случайная выборка всех LLM. Доверительный интервал условен на выбранной правке и тестовых примерах; он не учитывает вариативность повторных обучений и адаптивного поиска разных моделей. Сохраняется зависимость внутри общих шаблонов BLiMP; стратификация не превращает предложения в независимые новые грамматические конструкции. Длина токенизации влияет на суммарную вероятность. Наличие невнятных повторяющихся продолжений не позволяет делать вывод об общем улучшении генерации по одной сравнительной метрике. Итоговые ошибки нельзя использовать для нового подтверждающего подбора на том же test.

Инференс выполняется на CPU, float32, один поток, batch 64, official eager attention. Время измеряется после одинакового прогрева отдельно от загрузки и оценивания; это единичные наблюдения, не полноценный скоростной бенчмарк. Время подготовки правки в журнале охватывает программное применение и сериализацию, а не время рассуждений агента. Команды воспроизведения и версии зависимостей приведены в README.md и environment.json.

## Источники

1. Glint Research / CompactAI. Glimmer-1-Base. https://huggingface.co/Glint-Research/Glimmer-1-Base . Revision и хеши: sources.json.
2. Warstadt A., Parrish A., Liu H., Mohananey A., Peng W., Wang S.-F., Bowman S. R. BLiMP: The Benchmark of Linguistic Minimal Pairs for English. TACL, 2020, 8:377–392. https://doi.org/10.1162/tacl_a_00321 . Данные https://github.com/alexwarstadt/blimp , CC-BY 4.0. Наш поднабор и splitting являются изменением схемы оценки; авторы не подтверждали наши результаты.
3. Takehi R., Teoh J., Zhang A., Khattab O. Sparks of Weight-Space Programming. 2026. https://rikiyat.github.io/blog/weight-space-programming/ . Контекст исследования, не воспроизведение эксперимента с Qwen.
'''
    (paper/'ARTICLE_DRAFT_RU.md').write_text(draft)
    try:plots(paper,comp,selected)
    except ImportError:print('Tables/draft saved; install matplotlib for figures.')
    print('Paper artifacts:',paper)

def plots(paper,comparisons,selected):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for ext in ['pdf','svg','png']:fig.savefig(paper/f'{name}.{ext}',bbox_inches='tight',dpi=160)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.4))
    for i,row in enumerate(comparisons):
        lo,hi=row['paired_stratified_95ci_pp'];d=row['delta_pp']
        ax.errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o',capsize=4,color='#176b87' if i==0 else '#777777')
    ax.axvline(0,color='black',linestyle='--',linewidth=.8);ax.set_yticks(range(len(comparisons)),[r['model'] for r in comparisons]);ax.invert_yaxis()
    ax.set_xlabel('After − before pair accuracy (percentage points)');ax.set_title('BLiMP subset: paired 95% bootstrap intervals')
    discordant=selected['wrong_to_right']+selected['right_to_wrong']
    if discordant<25:
        fig.text(.12,-.12,f'Selected edit: only {discordant} discordant pairs; consult the exact paired sensitivity analysis.',fontsize=8)
    save(fig,'paired_changes')
    rows=selected['by_paradigm'];names=list(rows);x=np.arange(len(names));fig,ax=plt.subplots(figsize=(11,4))
    ax.bar(x-.18,[100*rows[k]['before_correct']/rows[k]['n'] for k in names],.36,label='Before')
    ax.bar(x+.18,[100*rows[k]['after_correct']/rows[k]['n'] for k in names],.36,label='Selected edit')
    ax.set_xticks(x,[n.replace('_',' ') for n in names],rotation=35,ha='right');ax.set_ylim(0,105)
    ax.set_ylabel('Pair accuracy (%)');ax.set_title('Held-out subset by paradigm; descriptive secondary results');ax.legend();save(fig,'paradigm_accuracy')

if __name__=='__main__':main()
