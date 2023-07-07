(require [language.string str]
         ["test_functions" f])

(let [counts {}]
  (loop [line (read-line)]
    (if (not (nil? line))
      (do
        (if (not (str/blank? line))
          (f/process-line counts line))
        (recur (read-line)))))
  (print counts))
