(def counts {})

(defn process-words
  [counts words]
  (let [numwords (count words)]
    (loop [i 0]
      (if (= i numwords)
        counts
        (let [word (nth words i)]
          (do
            (if (= false (str/blank? word))
              (let [curcount (get counts word 0)]
                (assoc counts word (+ curcount 1))))
            (recur (+ i 1))))))))
 
(defn process-line
  [counts line]
  (let [words (str/split (str/lower line))]
    (process-words counts words)))

(defn compare
  [a b]
  (> (nth a 1) (nth b 1)))

(loop [line (read-line)]
  (if (= nil line)
    nil
    (do
      (if (= false (str/blank? line))
        (process-line counts line))
      (recur (read-line)))))

(let [sortedlist (sort compare (pairs counts))
      numitems (count sortedlist)]
  (loop [index 0]
    (if (< index numitems)
      (let [entry (nth sortedlist index)]
        (do
          (print (nth entry 0))
          (print " ")
          (print (nth entry 1))
          (print "\n")
          (recur (+ index 1)))))))
