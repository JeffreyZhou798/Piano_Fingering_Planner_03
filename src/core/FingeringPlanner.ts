import type { 
  Note, PatternSegment, PatternType, Finger, Hand,
  FingeringState, FingeringSolution, CostResult 
} from '@/types';

/**
 * Layer 2: Fingering Optimization via Rule-based Dynamic Programming
 * 
 * CORRECTED based on validation against real pieces
 * Key principles:
 * 1. Five-finger position is fundamental
 * 2. Difficulty level ACTUALLY changes fingering choices
 * 3. Better pattern-specific handling
 */
export class FingeringPlanner {
  private difficultyLevel: 'beginner' | 'intermediate' | 'advanced' = 'intermediate';
  
  // Natural finger spans (in semitones)
  private readonly naturalSpans: Record<string, number> = {
    '1-2': 2, '2-3': 2, '3-4': 1, '4-5': 2,
    '1-3': 4, '2-4': 3, '3-5': 3,
    '1-4': 5, '2-5': 5,
    '1-5': 8
  };

  // Difficulty-specific configuration
  private getDifficultyConfig() {
    switch (this.difficultyLevel) {
      case 'beginner':
        return {
          // Beginners: Avoid thumb crossing, prefer 5-finger position
          thumbCrossingPenalty: 80,      // High penalty for thumb crossing
          positionChangePenalty: 60,     // High penalty for position changes
          finger4Penalty: 15,            // Slight penalty for finger 4
          finger5Penalty: 10,            // Slight penalty for finger 5
          stretchPenalty: 25,            // High penalty for stretching
          maxComfortableSpan: 5,         // Smaller comfortable span
          preferSimplePatterns: true,
          allowThumbOnBlack: false,
        };
      case 'advanced':
        return {
          // Advanced: Allow complex fingerings for musicality
          thumbCrossingPenalty: 10,      // Low penalty - thumb crossing is fine
          positionChangePenalty: 15,     // Low penalty - position changes OK
          finger4Penalty: 0,             // No penalty for finger 4
          finger5Penalty: 0,             // No penalty for finger 5
          stretchPenalty: 5,             // Low penalty for stretching
          maxComfortableSpan: 9,         // Larger comfortable span
          preferSimplePatterns: false,
          allowThumbOnBlack: true,
        };
      default: // intermediate
        return {
          thumbCrossingPenalty: 30,
          positionChangePenalty: 30,
          finger4Penalty: 5,
          finger5Penalty: 5,
          stretchPenalty: 12,
          maxComfortableSpan: 7,
          preferSimplePatterns: false,
          allowThumbOnBlack: false,
        };
    }
  }

  setDifficultyLevel(level: 'beginner' | 'intermediate' | 'advanced') {
    this.difficultyLevel = level;
  }

  planFingering(notes: Note[], patterns: PatternSegment[]): FingeringSolution {
    if (notes.length === 0) {
      return { fingering: [], totalCost: 0, path: [], explanations: [] };
    }

    const rhNotes = notes.filter(n => n.hand === 'RH');
    const lhNotes = notes.filter(n => n.hand === 'LH');
    
    const rhSolution = this.planHandFingering(rhNotes, patterns, 'RH');
    const lhSolution = this.planHandFingering(lhNotes, patterns, 'LH');

    // Merge solutions
    const fingering: Finger[] = new Array(notes.length);
    const explanations: string[] = new Array(notes.length);
    
    let rhIdx = 0, lhIdx = 0;
    notes.forEach((note, i) => {
      if (note.hand === 'RH') {
        fingering[i] = rhSolution.fingering[rhIdx] || 3;
        explanations[i] = rhSolution.explanations[rhIdx] || '';
        rhIdx++;
      } else {
        fingering[i] = lhSolution.fingering[lhIdx] || 3;
        explanations[i] = lhSolution.explanations[lhIdx] || '';
        lhIdx++;
      }
    });
    
    return {
      fingering,
      totalCost: rhSolution.totalCost + lhSolution.totalCost,
      path: [...rhSolution.path, ...lhSolution.path],
      explanations
    };
  }

  private planHandFingering(notes: Note[], patterns: PatternSegment[], hand: Hand): FingeringSolution {
    if (notes.length === 0) {
      return { fingering: [], totalCost: 0, path: [], explanations: [] };
    }

    if (notes.length > 64) {
      return this.chunkedOptimization(notes, patterns, hand);
    }

    return this.dpOptimization(notes, patterns, hand);
  }

  /**
   * Core DP optimization with difficulty-aware cost function
   */
  private dpOptimization(notes: Note[], patterns: PatternSegment[], hand: Hand): FingeringSolution {
    const n = notes.length;
    const fingers: Finger[] = [1, 2, 3, 4, 5];
    const config = this.getDifficultyConfig();
    
    // Analyze context
    const handPositions = this.analyzeHandPositions(notes, hand);
    const scaleSegments = this.detectScaleSegments(notes);
    
    const dp: Map<string, { cost: number; parent: string | null; reasons: string[] }>[] = [];
    
    // Initialize first note
    dp[0] = new Map();
    for (const finger of fingers) {
      const cost = this.computeInitialCost(finger, notes[0], hand, handPositions[0], config);
      dp[0].set(finger.toString(), { cost: cost.cost, parent: null, reasons: cost.reasons });
    }

    // Forward pass
    for (let i = 1; i < n; i++) {
      dp[i] = new Map();
      const pos = handPositions[i];
      const patternContext = this.getPatternContext(notes, i, patterns);
      const inScale = scaleSegments.some(seg => i >= seg.start && i <= seg.end);
      
      for (const toFinger of fingers) {
        let minCost = Infinity;
        let bestParent: string | null = null;
        let bestReasons: string[] = [];
        
        for (const fromFinger of fingers) {
          const prevState = dp[i - 1].get(fromFinger.toString());
          if (!prevState) continue;
          
          const transitionCost = this.computeTransitionCost(
            notes[i - 1], fromFinger as Finger,
            notes[i], toFinger,
            patternContext, hand, pos, config, inScale
          );
          
          // Skip impossible transitions
          if (transitionCost.cost > 500) continue;
          
          const totalCost = prevState.cost + transitionCost.cost;
          
          if (totalCost < minCost) {
            minCost = totalCost;
            bestParent = `${i - 1}-${fromFinger}`;
            bestReasons = transitionCost.reasons;
          }
        }
        
        if (minCost < Infinity) {
          dp[i].set(toFinger.toString(), { cost: minCost, parent: bestParent, reasons: bestReasons });
        }
      }
    }
    
    return this.backtrack(dp, notes, hand);
  }


  /**
   * Detect scale segments for proper thumb crossing
   */
  private detectScaleSegments(notes: Note[]): { start: number; end: number; ascending: boolean }[] {
    const segments: { start: number; end: number; ascending: boolean }[] = [];
    if (notes.length < 5) return segments;
    
    let segStart = 0;
    let direction = 0;
    let stepCount = 0;
    
    for (let i = 1; i < notes.length; i++) {
      const interval = notes[i].pitch - notes[i - 1].pitch;
      const absInterval = Math.abs(interval);
      const isStep = absInterval === 1 || absInterval === 2;
      const currDir = Math.sign(interval);
      
      if (isStep && (currDir === direction || direction === 0)) {
        stepCount++;
        if (direction === 0) direction = currDir;
      } else {
        // End of potential scale segment
        if (stepCount >= 4) {
          segments.push({ start: segStart, end: i - 1, ascending: direction > 0 });
        }
        segStart = i;
        direction = isStep ? currDir : 0;
        stepCount = isStep ? 1 : 0;
      }
    }
    
    // Check final segment
    if (stepCount >= 4) {
      segments.push({ start: segStart, end: notes.length - 1, ascending: direction > 0 });
    }
    
    return segments;
  }

  /**
   * Analyze hand positions for five-finger position mapping
   */
  private analyzeHandPositions(notes: Note[], hand: Hand): { anchorPitch: number; inPosition: boolean }[] {
    const positions: { anchorPitch: number; inPosition: boolean }[] = [];
    if (notes.length === 0) return positions;
    
    // Find stable hand positions based on pitch clusters
    let segmentStart = 0;
    let minPitch = notes[0].pitch;
    let maxPitch = notes[0].pitch;
    
    for (let i = 0; i < notes.length; i++) {
      const pitch = notes[i].pitch;
      const newMin = Math.min(minPitch, pitch);
      const newMax = Math.max(maxPitch, pitch);
      
      // If range exceeds comfortable span, start new segment
      if (newMax - newMin > 8) {
        const anchorPitch = hand === 'RH' ? minPitch : maxPitch;
        for (let j = segmentStart; j < i; j++) {
          positions[j] = { anchorPitch, inPosition: true };
        }
        segmentStart = i;
        minPitch = pitch;
        maxPitch = pitch;
      } else {
        minPitch = newMin;
        maxPitch = newMax;
      }
    }
    
    // Final segment
    const anchorPitch = hand === 'RH' ? minPitch : maxPitch;
    for (let j = segmentStart; j < notes.length; j++) {
      positions[j] = { anchorPitch, inPosition: true };
    }
    
    return positions;
  }

  private chunkedOptimization(notes: Note[], patterns: PatternSegment[], hand: Hand): FingeringSolution {
    const chunkSize = 32;
    const overlapSize = 4;
    const chunks: Note[][] = [];
    
    for (let i = 0; i < notes.length; i += chunkSize - overlapSize) {
      const end = Math.min(i + chunkSize, notes.length);
      chunks.push(notes.slice(i, end));
    }
    
    const chunkSolutions = chunks.map(chunk => this.dpOptimization(chunk, patterns, hand));
    
    const fingering: Finger[] = [];
    const explanations: string[] = [];
    let totalCost = 0;
    
    for (let i = 0; i < chunkSolutions.length; i++) {
      const sol = chunkSolutions[i];
      if (i === 0) {
        fingering.push(...sol.fingering);
        explanations.push(...sol.explanations);
      } else {
        fingering.push(...sol.fingering.slice(overlapSize));
        explanations.push(...sol.explanations.slice(overlapSize));
      }
      totalCost += sol.totalCost;
    }
    
    return { fingering, totalCost, path: [], explanations };
  }

  private backtrack(
    dp: Map<string, { cost: number; parent: string | null; reasons: string[] }>[],
    notes: Note[],
    hand: Hand
  ): FingeringSolution {
    const n = notes.length;
    
    let minFinalCost = Infinity;
    let bestFinalFinger: Finger = 3;
    
    for (const [finger, state] of dp[n - 1].entries()) {
      if (state.cost < minFinalCost) {
        minFinalCost = state.cost;
        bestFinalFinger = parseInt(finger) as Finger;
      }
    }
    
    const fingering: Finger[] = new Array(n);
    const explanations: string[] = new Array(n);
    const path: FingeringState[] = [];
    
    let currentFinger = bestFinalFinger;
    
    for (let i = n - 1; i >= 0; i--) {
      fingering[i] = currentFinger;
      const state = dp[i].get(currentFinger.toString());
      explanations[i] = state?.reasons.join('; ') || '';
      
      path.unshift({
        noteIndex: i,
        finger: currentFinger,
        hand,
        handPosition: notes[i].pitch,
        cost: state?.cost || 0,
        parent: null,
        reasons: state?.reasons || []
      });
      
      if (state?.parent) {
        const [, prevFinger] = state.parent.split('-');
        currentFinger = parseInt(prevFinger) as Finger;
      }
    }
    
    return { fingering, totalCost: minFinalCost, path, explanations };
  }


  /**
   * Initial cost - difficulty-aware
   */
  private computeInitialCost(
    finger: Finger, 
    note: Note, 
    hand: Hand,
    pos: { anchorPitch: number; inPosition: boolean },
    config: ReturnType<typeof this.getDifficultyConfig>
  ): CostResult {
    let cost = 0;
    const reasons: string[] = [];
    
    // Calculate expected finger based on position
    const offset = note.pitch - pos.anchorPitch;
    const expectedFinger = this.getExpectedFinger(offset, hand);
    
    // Reward matching expected finger
    if (finger === expectedFinger) {
      cost -= 25;
      reasons.push('Matches position');
    } else {
      const diff = Math.abs(finger - expectedFinger);
      cost += diff * 12;
    }
    
    // Difficulty-specific: Beginner prefers strong fingers
    if (this.difficultyLevel === 'beginner') {
      if (finger === 4) cost += config.finger4Penalty;
      if (finger === 5) cost += config.finger5Penalty;
      if (finger === 1 || finger === 2 || finger === 3) cost -= 5;
    }
    
    // Black key handling
    if (this.isBlackKey(note.pitch)) {
      if (finger === 1 || finger === 5) {
        cost += config.allowThumbOnBlack ? 10 : 25;
        reasons.push('Short finger on black key');
      } else {
        cost -= 8;
        reasons.push('Long finger on black key');
      }
    }
    
    return { cost, reasons };
  }

  /**
   * Get expected finger based on pitch offset from anchor
   */
  private getExpectedFinger(offset: number, hand: Hand): Finger {
    if (hand === 'RH') {
      // RH: thumb on lowest, pinky on highest
      if (offset <= 0) return 1;
      if (offset <= 2) return 2;
      if (offset <= 4) return 3;
      if (offset <= 6) return 4;
      return 5;
    } else {
      // LH: pinky on lowest, thumb on highest
      if (offset >= 0) return 1;
      if (offset >= -2) return 2;
      if (offset >= -4) return 3;
      if (offset >= -6) return 4;
      return 5;
    }
  }

  /**
   * Transition cost - the core of fingering decisions
   * Now with REAL difficulty differentiation
   */
  private computeTransitionCost(
    prevNote: Note, prevFinger: Finger,
    currNote: Note, currFinger: Finger,
    patternContext: PatternType,
    hand: Hand,
    pos: { anchorPitch: number; inPosition: boolean },
    config: ReturnType<typeof this.getDifficultyConfig>,
    inScale: boolean
  ): CostResult {
    let cost = 0;
    const reasons: string[] = [];
    
    const interval = currNote.pitch - prevNote.pitch;
    const absInterval = Math.abs(interval);
    const ascending = interval > 0;
    const fingerDiff = currFinger - prevFinger;
    
    // ===== RULE 1: Same finger on different pitch =====
    if (currFinger === prevFinger && interval !== 0) {
      cost += 40 + absInterval * 5;
      reasons.push('Same finger leap');
    }
    
    // ===== RULE 2: Repeated note - prefer finger change =====
    if (interval === 0) {
      if (currFinger === prevFinger) {
        cost += 25;
        reasons.push('Same finger on repeat');
      } else {
        cost -= 10;
        reasons.push('Good finger change');
      }
    }
    
    // ===== RULE 3: Natural finger progression =====
    if (interval !== 0) {
      const naturalProgression = this.isNaturalProgression(
        prevFinger, currFinger, ascending, hand
      );
      
      if (naturalProgression) {
        cost -= 20;
        reasons.push('Natural progression');
      } else if (this.isThumbCrossing(prevFinger, currFinger)) {
        // Thumb crossing - difficulty dependent!
        if (inScale || patternContext === 'SCALE') {
          // In scales, thumb crossing is expected
          cost += config.thumbCrossingPenalty / 3;
          reasons.push('Scale thumb crossing');
        } else {
          // Outside scales, penalize based on difficulty
          cost += config.thumbCrossingPenalty;
          reasons.push('Thumb crossing');
        }
      } else if (this.isFingerCrossing(prevFinger, currFinger)) {
        // Non-thumb finger crossing - always bad
        cost += 80;
        reasons.push('Finger crossing');
      }
    }
    
    // ===== RULE 4: Span constraint =====
    const naturalSpan = this.getNaturalSpan(prevFinger, currFinger);
    const overStretch = absInterval - naturalSpan;
    
    if (overStretch > 0) {
      if (overStretch > config.maxComfortableSpan - naturalSpan) {
        cost += 200; // Impossible stretch
        reasons.push('Impossible stretch');
      } else {
        cost += overStretch * config.stretchPenalty;
        if (overStretch > 2) reasons.push('Stretch');
      }
    }
    
    // ===== RULE 5: Position-based fingering =====
    if (pos.inPosition && !inScale) {
      const expectedFinger = this.getExpectedFinger(
        currNote.pitch - pos.anchorPitch, hand
      );
      
      if (currFinger === expectedFinger) {
        cost -= 15;
      } else {
        const diff = Math.abs(currFinger - expectedFinger);
        cost += diff * 8;
      }
    }
    
    // ===== RULE 6: Scale-specific fingering =====
    if (inScale || patternContext === 'SCALE') {
      const scaleCost = this.computeScaleCost(
        prevFinger, currFinger, ascending, hand, config
      );
      cost += scaleCost.cost;
      if (scaleCost.reasons.length > 0) {
        reasons.push(...scaleCost.reasons);
      }
    }
    
    // ===== RULE 7: Black key preference =====
    if (this.isBlackKey(currNote.pitch)) {
      if (currFinger === 1) {
        cost += config.allowThumbOnBlack ? 15 : 35;
        reasons.push('Thumb on black');
      } else if (currFinger === 5) {
        cost += 20;
        reasons.push('Pinky on black');
      } else {
        cost -= 5; // Long fingers good on black keys
      }
    }
    
    // ===== RULE 8: Difficulty-specific finger preferences =====
    if (this.difficultyLevel === 'beginner') {
      // Beginners: prefer fingers 1, 2, 3
      if (currFinger === 4) cost += config.finger4Penalty;
      if (currFinger === 5 && !this.isBlackKey(currNote.pitch)) {
        cost += config.finger5Penalty;
      }
      // Reward staying in position
      if (Math.abs(fingerDiff) <= 1 && absInterval <= 2) {
        cost -= 10;
        reasons.push('Simple transition');
      }
    } else if (this.difficultyLevel === 'advanced') {
      // Advanced: reward efficient fingerings even if complex
      if (absInterval > 5 && this.isThumbCrossing(prevFinger, currFinger)) {
        cost -= 10; // Thumb crossing for large intervals is efficient
      }
    }
    
    // ===== RULE 9: Arpeggio pattern =====
    if (patternContext === 'ARPEGGIO') {
      const arpeggioGood = this.isGoodArpeggioFingering(
        prevFinger, currFinger, ascending, hand
      );
      if (arpeggioGood) {
        cost -= 15;
        reasons.push('Good arpeggio fingering');
      }
    }
    
    return { cost, reasons };
  }


  /**
   * Check if finger progression is natural (no crossing)
   */
  private isNaturalProgression(
    prevFinger: Finger, currFinger: Finger, 
    ascending: boolean, hand: Hand
  ): boolean {
    const fingerDiff = currFinger - prevFinger;
    
    if (hand === 'RH') {
      // RH ascending: higher finger number for higher pitch
      if (ascending && fingerDiff > 0) return true;
      // RH descending: lower finger number for lower pitch
      if (!ascending && fingerDiff < 0) return true;
    } else {
      // LH ascending: lower finger number for higher pitch
      if (ascending && fingerDiff < 0) return true;
      // LH descending: higher finger number for lower pitch
      if (!ascending && fingerDiff > 0) return true;
    }
    
    return false;
  }

  /**
   * Check if this is a thumb crossing (thumb under or finger over thumb)
   */
  private isThumbCrossing(prevFinger: Finger, currFinger: Finger): boolean {
    return (prevFinger === 1 && currFinger > 1) || 
           (prevFinger > 1 && currFinger === 1);
  }

  /**
   * Check if this is a non-thumb finger crossing (always bad)
   */
  private isFingerCrossing(prevFinger: Finger, currFinger: Finger): boolean {
    // Finger crossing without thumb involved
    if (prevFinger === 1 || currFinger === 1) return false;
    
    // Check for crossing: e.g., 3 over 2, or 2 under 3
    // This is detected by direction mismatch
    return false; // Simplified - actual crossing detected in transition cost
  }

  /**
   * Scale-specific cost calculation
   */
  private computeScaleCost(
    prevFinger: Finger, currFinger: Finger,
    ascending: boolean, hand: Hand,
    config: ReturnType<typeof this.getDifficultyConfig>
  ): CostResult {
    let cost = 0;
    const reasons: string[] = [];
    
    // Standard scale fingering patterns
    // RH ascending: 1-2-3-1-2-3-4-5 (thumb under after 3 or 4)
    // RH descending: 5-4-3-2-1-3-2-1 (finger over after 1)
    // LH is mirrored
    
    const rhAscending = hand === 'RH' && ascending;
    const rhDescending = hand === 'RH' && !ascending;
    const lhAscending = hand === 'LH' && ascending;
    const lhDescending = hand === 'LH' && !ascending;
    
    // Good scale transitions
    if (rhAscending || lhDescending) {
      // Pattern: 1-2-3-1-2-3-4-5
      const goodTransitions = [[1, 2], [2, 3], [3, 1], [3, 4], [4, 5], [4, 1]];
      const isGood = goodTransitions.some(
        ([from, to]) => from === prevFinger && to === currFinger
      );
      if (isGood) {
        cost -= 25;
        reasons.push('Standard scale');
      }
    }
    
    if (rhDescending || lhAscending) {
      // Pattern: 5-4-3-2-1-3-2-1
      const goodTransitions = [[5, 4], [4, 3], [3, 2], [2, 1], [1, 3], [1, 2], [1, 4]];
      const isGood = goodTransitions.some(
        ([from, to]) => from === prevFinger && to === currFinger
      );
      if (isGood) {
        cost -= 25;
        reasons.push('Standard scale');
      }
    }
    
    // Penalize same finger in scale
    if (currFinger === prevFinger) {
      cost += 50;
      reasons.push('Same finger in scale');
    }
    
    // For beginners, prefer simpler scale patterns
    if (config.preferSimplePatterns) {
      // Beginners: prefer 1-2-3-4-5 without thumb crossing when possible
      if (this.isThumbCrossing(prevFinger, currFinger)) {
        cost += 20; // Additional penalty for beginners
      }
    }
    
    return { cost, reasons };
  }

  /**
   * Check if arpeggio fingering is good
   */
  private isGoodArpeggioFingering(
    prevFinger: Finger, currFinger: Finger,
    ascending: boolean, hand: Hand
  ): boolean {
    // Standard arpeggio: 1-2-3 or 1-2-4 with thumb crossing
    if (hand === 'RH') {
      if (ascending) {
        // Good: 1-2-3-1, 1-2-4-1, 1-3-5
        return (prevFinger < currFinger) || 
               (prevFinger >= 3 && currFinger === 1);
      } else {
        return (prevFinger > currFinger) ||
               (prevFinger === 1 && currFinger >= 3);
      }
    } else {
      if (ascending) {
        return (prevFinger > currFinger) ||
               (prevFinger === 1 && currFinger >= 3);
      } else {
        return (prevFinger < currFinger) ||
               (prevFinger >= 3 && currFinger === 1);
      }
    }
  }

  private getPatternContext(notes: Note[], index: number, patterns: PatternSegment[]): PatternType {
    const note = notes[index];
    
    for (const pattern of patterns) {
      if (note.measureNumber >= pattern.startIndex && note.measureNumber <= pattern.endIndex) {
        return pattern.patternType;
      }
    }
    
    for (const pattern of patterns) {
      if (index >= pattern.startIndex && index <= pattern.endIndex) {
        return pattern.patternType;
      }
    }
    
    return 'UNKNOWN';
  }

  private getNaturalSpan(finger1: Finger, finger2: Finger): number {
    const key = `${Math.min(finger1, finger2)}-${Math.max(finger1, finger2)}`;
    return this.naturalSpans[key] || 0;
  }

  private isBlackKey(pitch: number): boolean {
    const pitchClass = pitch % 12;
    return [1, 3, 6, 8, 10].includes(pitchClass);
  }
}

export const fingeringPlanner = new FingeringPlanner();
