# Custom Data Pipeline

### Motivation
    - Existing datasets have been combined together haphazardly, which may lead to unknown data leakage and contamination.
    - Some used datasets have been collected under extenuating circumstances that may bias or affect the data (data collected from parkinsons patients).
    - Some data is biased to cultures (a lot of french songs in one), and some is not even labelling songs (audiobooks included accidently, biasing data to certain emotions).
    - Classes are represented unevenly, however this can also be fixed in post processing with class weighting etc.

### Plan
    - Similar approach of using spotify web api to identify songs and extract names.
    - We then scrape the web for reviews of the song using sources listed below.
    - We then run our text sentiment model (GoRoBERTa for now), using it to classify the song into GEMS-9.

### Notes
    - We plan on using one of our team member's spotify library for now, as they can better attest to classification accuracy since they are familiar with the song personally.
    - Potentially only use highly rated reviews, as those are more likely to speak to the song's emotional aspects.
    - Consider an ensemble technique. Since our previous model was primarily focused on song emotion based on sound exclusively, and this new one may be more lyrically based (due to lyrical analysis in reviews), we could stack the two models to produce more robust classifications.

    PROS:
    - This approach employs a well established practice in MIR of using text as a proxy signal for emotion classification. 
    - However, this is typically applied for lyrics, which can still lead to some loss of information, as lyrics do not necessarily mirror song emotion.
    - Thus, our approach actually provides an advantage, as reviews can more wholistically express the emotions invoked by a certain song.
    - Therefore, song reviews are a suitable medium for distant supervision for the problem of MIR.

    CONS:
    - Coverage can be uneven depending on song popularity, as popular songs will have more reviews and thus more robust classifications.
    - Reviews can be lengthy, thus, runtime may become a consideration.

### Resources
Dedicated music review sites
    - Rate Your Music   (album level reviews only, still shows promise as reviews are long, and songs "list" names can be used to classify instead)
    - Album of the Year (album level reviews only)
    - Pitchfork         (album and song reviews, but not great coverage)
    - AllMusic          (not great coverage, but has a great moods and themes tab, along with reviews)
    - Musicboard        (Both album and track reviews, mediocore coverage)

General platforms
    - Amazon Music
    - Apple Music
    - YouTube comments

Community/forum sources
    - Reddit
    - Last.fm           (Has per track breakdowns, unique emotion tags, and backstories)