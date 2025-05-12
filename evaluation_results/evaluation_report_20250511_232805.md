# Prompt Retrieval Model Evaluation Report
    
## Dataset Information
- Total examples: 532
- Evaluation date: 2025-05-11 23:28:05
- Dataset composition: {'positive_examples': 316, 'negative_examples': 216}

## Model Comparison

| Model | Precision@1 | Precision@3 | Precision@5 | MRR | NDCG@5 |
|-------|------------|-------------|-------------|-----|--------|
| all-MiniLM-L6-v2 | 0.547 | 0.483 | 0.459 | 0.193 | 0.216 |
| all-mpnet-base-v2 | 0.508 | 0.465 | 0.447 | 0.155 | 0.176 |
| multi-qa-mpnet-base-dot-v1 | 0.543 | 0.479 | 0.457 | 0.190 | 0.211 |

## Selected Model: all-MiniLM-L6-v2

### Selection Rationale
1. **Best Overall Performance**: all-MiniLM-L6-v2 achieved the highest Mean Reciprocal Rank (MRR) of 0.193, indicating better ranking of relevant results.
2. **Consistent Precision**: Maintains strong precision across different k values:
   - P@1: 0.547
   - P@3: 0.483
   - P@5: 0.459
3. **NDCG Performance**: Shows strong ranking quality with NDCG@5 of 0.216

### Example Queries and Results

#### Query: "Find a fantasy story about magic and potions"

1. **Score: 0.577**
   ```
   <Fantasy> </Fantasy> <Magic> </Magic> <The Accidental Alchemist> </The Accidental Alchemist> prompt A clumsy apprentice baker accidentally creates a potion that grants temporary, unpredictable magical abilities instead of a perfect sourdough starter.
   ```

2. **Score: 0.571**
   ```
   <Romance> </Romance> <Magic> </Magic> <The Love Potion Mishap> </The Love Potion Mishap> prompt Someone tries to use a love potion, but it accidentally affects the wrong person, leading to comical and complicated situations.
   ```

3. **Score: 0.498**
   ```
   <Fantasy> </Fantasy> <Coming of Age> </Coming of Age> <The Fading Magic> </The Fading Magic> prompt In a world where everyone has a small, personal magic, a teenager's magic starts to fade, and they must find out why before it's gone completely.
   ```

#### Query: "A horror story with a supernatural twist"

1. **Score: 0.569**
   ```
   <Horror> </Horror> <Supernatural> </Supernatural> <The Whispering Walls> </The Whispering Walls> prompt A family moves into an old house where the walls whisper secrets, but only the youngest child can understand them.
   ```

2. **Score: 0.560**
   ```
   <Horror> </Horror> <Supernatural> </Supernatural> <The Doll That Ages> </The Doll That Ages> prompt An antique doll in the attic seems to be aging, mirroring the appearance of someone in the family.
   ```

3. **Score: 0.542**
   ```
   <Mystery> </Mystery> <Supernatural> </Mystery> <The Haunted Bookstore </The Haunted Bookstore> prompt A bookstore is haunted by a playful but mischievous ghost who rearranges books to send messages or play pranks on customers.
   ```

#### Query: "Romance in a bookstore"

1. **Score: 0.807**
   ```
   <Romance> </Romance> <Love> </Love> <The Bookstore Meet-Cute> </The Bookstore Meet-Cute> prompt Two people keep reaching for the same obscure book in a dusty, independent bookstore.
   ```

2. **Score: 0.531**
   ```
   <Mystery> </Mystery> <Supernatural> </Mystery> <The Haunted Bookstore </The Haunted Bookstore> prompt A bookstore is haunted by a playful but mischievous ghost who rearranges books to send messages or play pranks on customers.
   ```

3. **Score: 0.444**
   ```
   <Romance> </Romance> <Friendship> </Romance> <From Pen Pals to Passion> </From Pen Pals to Passion> prompt Two people who have been anonymous pen pals for years, sharing their deepest secrets, finally decide to meet.
   ```

#### Query: "Science fiction about time travel"

1. **Score: 0.623**
   ```
   <Science Fiction> </Science Fiction> <Time Travel> </Time Travel> <The Accidental Tourist> </The Accidental Tourist> prompt Someone accidentally activates a faulty time machine and ends up in a completely random, inconvenient historical period with only the clothes on their back and a smartphone with 10% battery.
   ```

2. **Score: 0.585**
   ```
   <Science Fiction> </Science Fiction> <Time Travel> </Science Fiction> <The Message from the Future Self> </The Message from the Future Self> prompt You receive a cryptic, urgent message from your future self, warning you about a decision you're about to make.
   ```

3. **Score: 0.419**
   ```
   <Science Fiction> </Science Fiction> <Dystopian> </Science Fiction> <The Memory Market> </The Memory Market> prompt In a future where memories can be bought and sold, someone buys a set of happy childhood memories, only to find they come with unexpected and dangerous echoes.
   ```

## Evaluation Metrics Explained
- **Precision@k**: The proportion of relevant results in the top k retrieved items
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank of the first relevant result
- **NDCG@5 (Normalized Discounted Cumulative Gain)**: Measures ranking quality considering position

## Next Steps and Recommendations
1. Consider fine-tuning the selected model on domain-specific data
2. Implement post-processing rules to boost genre tag matches
3. Add relevance feedback mechanism for continuous improvement
