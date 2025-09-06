import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
import { Observable, BehaviorSubject, timer, EMPTY } from 'rxjs';
import { catchError, retry, switchMap, takeUntil } from 'rxjs/operators';
import { environment } from '../../environments/environment';

export interface HealthStatus {
  status: 'healthy' | 'loading' | 'error';
  models: {
    image_model: 'ready' | 'loading' | 'error';
    content_model: 'ready' | 'loading' | 'error';
  };
  first_job_completed: boolean;
  message: string;
}

@Injectable({
  providedIn: 'root'
})
export class HealthService {
  private readonly apiUrl = `${environment.apiUrl}/health`;
  private healthStatusSubject = new BehaviorSubject<HealthStatus | null>(null);
  private isCheckingSubject = new BehaviorSubject<boolean>(false);
  
  public readonly healthStatus$ = this.healthStatusSubject.asObservable();
  public readonly isChecking$ = this.isCheckingSubject.asObservable();

  constructor(private http: HttpClient) {}

  /**
   * Perform a single health check
   */
  checkHealth(): Observable<HealthStatus> {
    // Add header to bypass ngrok browser warning
    const headers = new HttpHeaders({
      'ngrok-skip-browser-warning': 'true'
    });
    
    return this.http.get<HealthStatus>(this.apiUrl, { headers }).pipe(
      catchError((error: HttpErrorResponse) => {
        console.error('Health check failed:', error);
        
        // Handle different error scenarios
        if (error.status === 503) {
          // Service unavailable - models still loading
          return [error.error as HealthStatus];
        } else if (error.status === 0) {
          // Network error - server not reachable
          return [{
            status: 'error' as const,
            models: {
              image_model: 'error' as const,
              content_model: 'error' as const
            },
            first_job_completed: false,
            message: 'Cannot connect to server'
          }];
        } else {
          // Other server errors
          return [{
            status: 'error' as const,
            models: {
              image_model: 'error' as const,
              content_model: 'error' as const
            },
            first_job_completed: false,
            message: `Health check failed: ${error.message}`
          }];
        }
      })
    );
  }

  /**
   * Start continuous health monitoring until models are ready
   */
  startHealthMonitoring(): Observable<HealthStatus> {
    this.isCheckingSubject.next(true);
    
    return timer(0, 3000).pipe( // Check immediately, then every 3 seconds
      switchMap(() => this.checkHealth()),
      retry(3), // Retry up to 3 times on failure
      takeUntil(
        this.healthStatusSubject.pipe(
          // Stop monitoring when status becomes 'healthy'
          switchMap(status => status?.status === 'healthy' ? [true] : EMPTY)
        )
      )
    );
  }

  /**
   * Check health once and update the status
   */
  async performHealthCheck(): Promise<HealthStatus> {
    this.isCheckingSubject.next(true);
    
    try {
      const status = await this.checkHealth().toPromise();
      this.healthStatusSubject.next(status!);
      
      if (status!.status === 'healthy') {
        this.isCheckingSubject.next(false);
      }
      
      return status!;
    } catch (error) {
      const errorStatus: HealthStatus = {
        status: 'error',
        models: {
          image_model: 'error',
          content_model: 'error'
        },
        first_job_completed: false,
        message: 'Health check failed'
      };
      
      this.healthStatusSubject.next(errorStatus);
      this.isCheckingSubject.next(false);
      return errorStatus;
    }
  }

  /**
   * Wait for models to be ready with automatic retry
   */
  async waitForModelsReady(timeoutMs: number = 120000): Promise<HealthStatus> {
    const startTime = Date.now();
    
    return new Promise((resolve, reject) => {
      const subscription = this.startHealthMonitoring().subscribe({
        next: (status) => {
          this.healthStatusSubject.next(status);
          
          if (status.status === 'healthy') {
            this.isCheckingSubject.next(false);
            subscription.unsubscribe();
            resolve(status);
          } else if (Date.now() - startTime > timeoutMs) {
            this.isCheckingSubject.next(false);
            subscription.unsubscribe();
            reject(new Error('Health check timeout - models took too long to load'));
          }
        },
        error: (error) => {
          this.isCheckingSubject.next(false);
          subscription.unsubscribe();
          reject(error);
        }
      });
    });
  }

  /**
   * Get the current health status without making a new request
   */
  getCurrentStatus(): HealthStatus | null {
    return this.healthStatusSubject.value;
  }

  /**
   * Check if models are currently ready
   */
  areModelsReady(): boolean {
    const status = this.getCurrentStatus();
    return status?.status === 'healthy';
  }

  /**
   * Check if the first job has been completed (for dynamic loading times)
   */
  isFirstJobCompleted(): boolean {
    const status = this.getCurrentStatus();
    return status?.first_job_completed ?? false;
  }
}