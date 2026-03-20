import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    
    # determine the links
    link_to = corpus[page]
    proba_distri = {}  # the return variable 

    num_links = len(link_to)
    if num_links == 0:
        for c in corpus:
            proba_distri[c] = 1 / len(corpus)
        return proba_distri

    random_jump = (1 - damping_factor) / len(corpus)

    for c in corpus:
        proba_distri[c] = random_jump

    for link in link_to:
        proba_distri[link] += (damping_factor / num_links)

    return proba_distri


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """    

    # init variable
    samples = {}
    # pick one start
    page = random.choice(list(corpus.keys()))
    samples[page] = samples.get(page, 0) + 1

    for i in range(n - 1):
        probability_distri = transition_model(corpus, page, damping_factor)
        item = list(probability_distri.keys())
        weights = list(probability_distri.values())
        page = random.choices(item, weights, k=1)[0]  # retrieve new page based on the current page
        samples[page] = samples.get(page, 0) + 1    

    total = sum(samples.values())
    samples_rank = {k: v/total for k, v in samples.items()}  # normalization
    return samples_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # init variables
    page_rank = {k: 1/len(corpus) for k in corpus}    
    damping_pr = (1 - damping_factor) / len(corpus)

    # pick one start
    # page = random.choice(list(corpus.keys()))

    while True:
        
        current_page_rank = copy.deepcopy(page_rank)
        # loop through all pages in corpus
        for page in corpus:

            the_other_factor = 0
            for the_other in corpus:                
                if len(corpus[the_other]) == 0: 
                    # handle page has no links at all. treat it as having links to all pages
                    the_other_pr = page_rank[the_other]
                    the_other_num_links = len(corpus)
                    the_other_factor += the_other_pr / the_other_num_links
                    continue                
                elif page not in corpus[the_other]:  # doen't have link to this page
                    continue

                the_other_pr = page_rank[the_other]
                the_other_num_links = len(corpus[the_other])
                the_other_factor += the_other_pr / the_other_num_links
            this_pr = damping_pr + damping_factor * the_other_factor
            current_page_rank[page] = this_pr

        # check if the page ranks change small enough
        diff = max(
            abs(current_page_rank[p] - page_rank[p])
            for p in page_rank
        )
        if diff <= 0.001:
            break
        # update the new set of page ranks
        page_rank = copy.deepcopy(current_page_rank) 
    return page_rank    


if __name__ == "__main__":
    main()
